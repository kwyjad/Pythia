from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".parq"}


def count_rows_csv(csv_path: Path) -> tuple[int | None, str | None]:
    try:
        total = 0
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for total, _ in enumerate(handle, start=1):
                pass
        if total == 0:
            return 0, None
        return max(total - 1, 0), None
    except Exception as exc:
        return None, f"failed to count CSV rows ({exc})"


def count_rows_parquet(parquet_path: Path) -> tuple[int | None, str | None]:
    errors: list[str] = []

    # Prefer DuckDB for speed and low memory usage.
    try:
        import duckdb  # type: ignore
        try:
            con = duckdb.connect()
        except Exception as exc:
            errors.append(f"duckdb connect failed ({exc})")
        else:
            try:
                rows = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [str(parquet_path)],
                ).fetchone()[0]
                return int(rows), None
            except Exception as exc:
                errors.append(f"duckdb count failed ({exc})")
            finally:
                try:
                    con.close()
                except Exception:
                    pass
    except ModuleNotFoundError as exc:
        errors.append(f"duckdb missing ({exc})")

    # Try PyArrow metadata if DuckDB path failed.
    try:
        import pyarrow.parquet as pq  # type: ignore
        try:
            pf = pq.ParquetFile(str(parquet_path))
            md = pf.metadata
            if md is not None:
                return int(md.num_rows), None
        except Exception as exc:
            errors.append(f"pyarrow failed ({exc})")
    except ModuleNotFoundError as exc:
        errors.append(f"pyarrow missing ({exc})")

    # Last resort: pandas
    try:
        import pandas as pd  # type: ignore
        try:
            frame = pd.read_parquet(parquet_path)
            return int(frame.shape[0]), None
        except Exception as exc:
            errors.append(f"pandas failed ({exc})")
    except ModuleNotFoundError as exc:
        errors.append(f"pandas missing ({exc})")

    message = "; ".join(errors) if errors else "unknown failure"
    return None, message


def iter_canonical_files(canonical_dir: Path) -> Iterable[Path]:
    for path in sorted(canonical_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize canonical CSV/Parquet files for diagnostics")
    parser.add_argument("--dir", dest="directory", required=True, help="Canonical directory to inspect")
    parser.add_argument("--out", required=True, help="Where to write the summary report")
    args = parser.parse_args(argv)

    canonical_dir = Path(args.directory)
    out_path = Path(args.out)

    lines: list[str] = ["== Canonical directory listing ==", f"directory: {canonical_dir}"]
    payload: dict[str, Any] = {
        "directory": str(canonical_dir),
        "exists": canonical_dir.is_dir(),
        "files": [],
    }

    if not canonical_dir.exists():
        lines.append("canonical directory missing; nothing to list")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
        return 0

    if not canonical_dir.is_dir():
        lines.append("canonical path exists but is not a directory")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
        return 0

    entries: list[dict[str, Any]] = []
    total_known_rows = 0
    unknown_counts = 0
    for path in iter_canonical_files(canonical_dir):
        rel = path.relative_to(canonical_dir)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows, error = count_rows_csv(path)
        else:
            rows, error = count_rows_parquet(path)

        size = path.stat().st_size
        entry: dict[str, Any] = {
            "path": str(rel),
            "rows": rows,
            "bytes": size,
            "type": suffix.lstrip("."),
        }
        if error:
            entry["error"] = error
        if rows is not None:
            total_known_rows += rows
            row_display = f"rows={rows}"
        else:
            unknown_counts += 1
            row_display = "rows=?"
        line = f"{rel} ({entry['type']}): {row_display} bytes={size}"
        if error:
            line += f" error={error}"
        lines.append(line)
        entries.append(entry)

    payload["files"] = entries
    payload["total_rows"] = total_known_rows
    payload["unknown_row_counts"] = unknown_counts

    if not entries:
        lines.append("no CSV/Parquet files found")
    else:
        lines.append(f"files: {len(entries)}")
        lines.append(f"total rows (excluding headers where applicable): {total_known_rows}")
        if unknown_counts:
            lines.append(f"files with unknown row count: {unknown_counts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
