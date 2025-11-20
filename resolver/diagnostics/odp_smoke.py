from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from resolver.db import duckdb_io
from resolver.ingestion import odp_series


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _normalize_db_url(db_path: str | Path) -> tuple[str, Path | None]:
    text_path = str(db_path)
    if text_path.lower().startswith("duckdb:///"):
        resolved = Path(text_path[len("duckdb:///") :]).expanduser().resolve()
        return text_path, resolved
    if "://" in text_path:
        try:
            return text_path, Path(text_path)
        except Exception:  # noqa: BLE001
            return text_path, None
    candidate_path = Path(text_path)
    resolved = candidate_path.expanduser().resolve()
    return f"duckdb:///{resolved.as_posix()}", resolved


def _collect_duckdb_stats(db_path: str | Path) -> Mapping[str, object]:
    db_url, resolved_path = _normalize_db_url(db_path)
    exists = resolved_path.exists() if resolved_path else False
    info: dict[str, object] = {
        "db_path": str(db_path),
        "db_url": db_url,
        "db_exists": exists,
        "table_exists": False,
        "row_count": 0,
        "coverage": [],
        "error": None,
    }
    if not exists:
        return info

    try:
        conn = duckdb_io.get_db(db_url)
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return info

    try:
        try:
            total_rows = conn.execute("SELECT COUNT(*) FROM odp_timeseries_raw").fetchone()[0]
            info["table_exists"] = True
            info["row_count"] = int(total_rows)
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
            return info

        if total_rows:
            coverage_rows: Sequence[tuple[str, str, str, str, int]] = conn.execute(
                """
                SELECT
                  iso3,
                  metric,
                  MIN(ym) AS first_ym,
                  MAX(ym) AS last_ym,
                  COUNT(*) AS n_rows
                FROM odp_timeseries_raw
                GROUP BY iso3, metric
                ORDER BY iso3, metric
                """
            ).fetchall()
            info["coverage"] = [
                {
                    "iso3": row[0],
                    "metric": row[1],
                    "first_ym": row[2],
                    "last_ym": row[3],
                    "rows": int(row[4]),
                }
                for row in coverage_rows
            ]
    finally:
        duckdb_io.close_db(conn)
    return info


def _list_artifacts(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        return []
    files: list[str] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(base_dir).as_posix())
    return files


def _render_coverage_table(entries: Iterable[Mapping[str, object]]) -> list[str]:
    rows = list(entries)
    if not rows:
        return ["(no rows in odp_timeseries_raw)"]
    lines = [
        "| iso3 | metric | rows | first_ym | last_ym |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in rows:
        lines.append(
            "| {iso3} | {metric} | {rows} | {first} | {last} |".format(
                iso3=entry.get("iso3") or "",  # type: ignore[arg-type]
                metric=entry.get("metric") or "",  # type: ignore[arg-type]
                rows=entry.get("rows", 0),
                first=entry.get("first_ym") or "",  # type: ignore[arg-type]
                last=entry.get("last_ym") or "",  # type: ignore[arg-type]
            )
        )
    return lines


def build_smoke_summary(
    *,
    config_path: str | Path,
    normalizers_path: str | Path,
    db_path: str | Path,
    diagnostics_dir: str | Path,
    rows: int | None,
    error: Exception | None,
    traceback_text: str | None,
    stats: odp_series.OdpPipelineStats | None = None,
) -> str:
    config = Path(config_path)
    normalizers = Path(normalizers_path)
    diag_dir = Path(diagnostics_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    duckdb_info = _collect_duckdb_stats(db_path)
    env_flags = {"ODP_JSON_NETWORK": os.getenv("ODP_JSON_NETWORK", "<unset>")}
    artifacts = _list_artifacts(diag_dir)

    lines: list[str] = ["# ODP smoke summary", ""]

    lines.extend(
        [
            "## Inputs",
            f"- Config path: {config} (exists: {_yes_no(config.exists())})",
            f"- Normalizers path: {normalizers} (exists: {_yes_no(normalizers.exists())})",
            f"- Config pages: {stats.config_pages if stats is not None else 'unknown'}",
            f"- ODP_JSON_NETWORK: {env_flags['ODP_JSON_NETWORK']}",
            "",
        ]
    )

    lines.extend(
        [
            "## Discovery & normalization",
            f"- Pages discovered: {stats.pages_discovered if stats is not None else 'unknown'}",
            f"- JSON links found: {stats.json_links_found if stats is not None else 'unknown'}",
            f"- JSON links matched: {stats.json_links_matched if stats is not None else 'unknown'}",
            f"- JSON links unmatched: {stats.json_links_unmatched if stats is not None else 'unknown'}",
            f"- Raw records total: {stats.raw_records_total if stats is not None else 'unknown'}",
            f"- Normalized rows total: {stats.normalized_rows_total if stats is not None else (rows if rows is not None else 'unknown')}",
        ]
    )

    lines.append("- Normalized rows per series:")
    if stats is not None and stats.normalized_rows_per_series:
        lines.append("  | source_id | rows |")
        lines.append("  | --- | --- |")
        for source_id, count in sorted(stats.normalized_rows_per_series.items()):
            lines.append(f"  | {source_id} | {count} |")
    else:
        lines.append("  - (no normalized rows)")

    lines.append("- Unmatched widget labels:")
    if stats is not None and stats.unmatched_labels:
        for label in sorted(stats.unmatched_labels):
            lines.append(f"  - {label}")
    else:
        lines.append("  - (none)")

    if stats is not None and stats.notes:
        lines.append("- Notes:")
        for note in sorted(stats.notes):
            lines.append(f"  - {note}")
    lines.append("")

    lines.extend(
        [
            "## DuckDB",
            f"- DuckDB path: {duckdb_info['db_path']}",
            f"- DuckDB exists: {_yes_no(bool(duckdb_info['db_exists']))}",
        ]
    )

    if duckdb_info["error"]:
        lines.append(f"- DuckDB error: {duckdb_info['error']}")
    elif duckdb_info["table_exists"]:
        lines.extend(
            [
                f"- odp_timeseries_raw present: {_yes_no(True)}",
                f"- odp_timeseries_raw total rows: {duckdb_info['row_count']}",
                "",
                "### odp_timeseries_raw coverage by iso3/metric",
                *_render_coverage_table(duckdb_info["coverage"]),
            ]
        )
    else:
        lines.append("- odp_timeseries_raw present: no")
    lines.append("")

    lines.extend(
        [
            "## Diagnostics artifacts",
            f"- diagnostics dir: {diag_dir} (exists: {_yes_no(diag_dir.exists())})",
        ]
    )
    if artifacts:
        for artifact in artifacts:
            lines.append(f"  - {artifact}")
    else:
        lines.append("  - (no artifacts found)")
    lines.append("")

    lines.append("## Result")
    if error:
        lines.append("- Outcome: FAILED")
        lines.append(f"- Error type: {type(error).__name__}")
        if str(error):
            lines.append(f"- Error message: {error}")
        if traceback_text:
            lines.extend(["", "```", traceback_text.strip(), "```"])
    else:
        lines.append("- Outcome: SUCCESS")
        lines.append(f"- Rows written: {rows if rows is not None else 0}")

    if not lines[-1].endswith("\n"):
        lines.append("")
    return "\n".join(lines)


def write_smoke_summary(markdown: str, path: str | Path) -> Path:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(markdown, encoding="utf-8")
    return summary_path


__all__ = ["build_smoke_summary", "write_smoke_summary"]
