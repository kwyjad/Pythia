from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle)
    except UnicodeDecodeError:
        with csv_path.open("r", encoding="latin-1", errors="ignore") as handle:
            total = sum(1 for _ in handle)
    if total == 0:
        return 0
    return max(total - 1, 0)


def collect(canonical_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    for path in sorted(canonical_dir.rglob("*.csv")):
        rel_path = path.relative_to(canonical_dir)
        rows = count_rows(path)
        size = path.stat().st_size
        lines.append(f"{rel_path}: rows={rows} bytes={size}")
        entries.append(
            {
                "path": str(rel_path),
                "rows": rows,
                "bytes": size,
            }
        )
    return lines, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize canonical CSV files for diagnostics")
    parser.add_argument("--dir", dest="directory", required=True, help="Canonical directory to inspect")
    parser.add_argument("--out", required=True, help="Where to write the summary report")
    args = parser.parse_args(argv)

    canonical_dir = Path(args.directory)
    out_path = Path(args.out)

    header = "== Canonical directory listing =="
    lines: list[str] = [header, f"directory: {canonical_dir}"]
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

    file_lines, entries = collect(canonical_dir)
    payload["files"] = entries
    total_rows = sum(item.get("rows", 0) for item in entries)
    payload["total_rows"] = total_rows

    if not file_lines:
        lines.append("no CSV files found")
    else:
        lines.append(f"files: {len(file_lines)}")
        lines.extend(file_lines)
        lines.append(f"total rows (excluding headers): {total_rows}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
