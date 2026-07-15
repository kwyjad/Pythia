# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Consolidated Resolver Update run summary for the GitHub Step Summary.

Reads ``connectors_report.jsonl`` (status/reason/duration per connector) and
the just-written resolver DuckDB (authoritative row/country counts), and
produces a clean Markdown summary:

  1. A one-line result verdict + per-phase roll-up + effective ingest windows.
  2. A **Problems & Warnings** section that surfaces every failed/skipped
     source, every "succeeded but wrote 0 rows" source, stale conflict-forecast
     vintages, a shrunken fewsnet_countries.json, and whole-country ACLED
     political attribution drops, with reasons.
  3. Per-phase tables with uniform, honestly-labelled columns.
  4. Conflict-forecast vintage table + CrisisWatch edition status.

Why the DB is the source of truth: the JSONL ``counts`` are inconsistent
across connector types — per-country sources report *countries* in
``written``, self-storing global sources report *rows* in ``fetched`` and a
nominal 1-2 in ``written``, and some (gdelt, nmme) report neither. Querying
the target table gives one uniform, trustworthy number. The per-run delta
is taken from the JSONL only where it is reliably a row count.

Usage:
    python -m pythia.tools.summarize_all_phases \
        --report diagnostics/ingestion/connectors_report.jsonl \
        --db data/resolver.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connector metadata: phase + authoritative DB target
# ---------------------------------------------------------------------------
# connector_id -> (phase_num, table, where_clause | None, iso_col | None)
# `table`/`where` drive the authoritative "Rows in DB" / "Countries" columns;
# publisher/source filters are verified exact strings from facts_resolved /
# conflict_forecasts. iso_col=None means the table has no country dimension.
_CONNECTORS: dict[str, tuple[int, Optional[str], Optional[str], Optional[str]]] = {
    # Phase 1 — Ground truth
    "acled_client":            (1, "facts_resolved", "publisher = 'ACLED'", "iso3"),
    "ifrc_go_client":          (1, "facts_resolved", "publisher = 'IFRC'", "iso3"),
    "idmc":                    (1, "facts_resolved", "publisher = 'IDMC'", "iso3"),
    "idmc_helix":              (1, "facts_resolved", "publisher = 'IDMC'", "iso3"),
    # Phase 2 — Resolution sources (facts_resolved)
    "fewsnet_ipc_population":  (2, "facts_resolved", "publisher = 'FEWS NET'", "iso3"),
    "ipc_api_population":      (2, "facts_resolved", "publisher = 'IPC'", "iso3"),
    "gdacs_population_exposed": (2, "facts_resolved", "publisher = 'GDACS / JRC'", "iso3"),
    # Phase 3 — Structured data (Pythia tables)
    "views_forecasts":         (3, "conflict_forecasts", "source = 'VIEWS'", "iso3"),
    "conflictforecast_forecasts": (3, "conflict_forecasts", "source = 'conflictforecast_org'", "iso3"),
    "acledcast_forecasts":     (3, "conflict_forecasts", "source = 'ACLED_CAST'", "iso3"),
    "acaps_inform_severity":   (3, "acaps_inform_severity", None, "iso3"),
    "acaps_risk_radar":        (3, "acaps_risk_radar", None, "iso3"),
    "acaps_daily_monitoring":  (3, "acaps_daily_monitoring", None, "iso3"),
    "acaps_humanitarian_access": (3, "acaps_humanitarian_access", None, "iso3"),
    "reliefweb_reports":       (3, "reliefweb_reports", None, "iso3"),
    "acled_political_events":  (3, "acled_political_events", None, "iso3"),
    "nmme_seasonal_forecasts": (3, "seasonal_forecasts", None, "iso3"),
    "gdelt_conflict_indicators": (3, "gdelt_conflict_indicators", None, "iso3"),
    "ipc_phases":              (3, "ipc_phases", None, "iso3"),
    # Phase 4 — Context sources
    "enso":                    (4, "enso_state", None, None),
    "seasonal_tc":             (4, "seasonal_tc_outlooks", None, None),
    "hdx_signals":             (4, "hdx_signals", None, "iso3"),
    "crisiswatch":             (4, "crisiswatch_entries", None, "iso3"),
}

_PHASE_NAMES = {
    1: "Phase 1 — Ground Truth (ACLED, IFRC, IDMC → facts_resolved)",
    2: "Phase 2 — Resolution Sources (FEWS NET, IPC, GDACS → facts_resolved)",
    3: "Phase 3 — Structured Data (Pythia tables)",
    4: "Phase 4 — Context Sources",
}

_STATUS_EMOJI = {"ok": "✅", "error": "❌", "skipped": "⏭️"}

# Conflict forecasts (VIEWS / ACLED CAST / conflictforecast.org) publish
# monthly; a vintage older than this is flagged (mirrors the connectors'
# own _STALENESS_WARN_DAYS and the inspect report threshold).
_CF_STALENESS_WARN_DAYS = 45

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _status_icon(status: str) -> str:
    return _STATUS_EMOJI.get(status, f"`{status}`")


def _fmt_duration(duration_ms: int) -> str:
    if not duration_ms:
        return "—"
    secs = duration_ms / 1000
    if secs >= 60:
        return f"{secs / 60:.1f}m"
    return f"{secs:.1f}s"


def _fmt_int(n: Any) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


# ---------------------------------------------------------------------------
# JSONL loader + per-run row extractor
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            LOG.warning("Skipping invalid JSONL line: %s", line[:120])
    return entries


def _rows_written_this_run(entry: dict) -> Optional[int]:
    """Rows this connector wrote THIS run, or None when not reliably a row count.

    The JSONL is inconsistent, so only trust fields that are unambiguously
    rows: self-storing sources stamp ``extras.{conflict_rows,total_facts,
    resolved_rows}``; direct-store P1/P4 connectors (no per-country loop, so
    no ``empty`` key) put rows in ``counts.written``. Per-country sources put
    *countries* in ``written`` (they carry an ``empty`` key) — return None so
    the caller shows the authoritative DB count / country count instead.
    """
    extras = entry.get("extras") or {}
    for key in ("conflict_rows", "total_facts", "resolved_rows"):
        val = extras.get(key)
        if isinstance(val, int):
            return val
    counts = entry.get("counts") or {}
    if "empty" in counts:  # per-country loop → written is a country count
        return None
    val = counts.get("written")
    return val if isinstance(val, int) else None


# ---------------------------------------------------------------------------
# DuckDB access (authoritative counts)
# ---------------------------------------------------------------------------


def _resolve_db_path(db_path: str | None) -> str:
    if db_path is None:
        db_path = (
            os.environ.get("PYTHIA_DB_URL")
            or os.environ.get("BACKFILL_DB_PATH")
            or "data/resolver.duckdb"
        )
    if db_path.startswith("duckdb:///"):
        db_path = db_path[len("duckdb:///"):]
    return db_path


def _open_db(db_path: str | None):
    path = _resolve_db_path(db_path)
    if not Path(path).exists():
        LOG.warning("Resolver DB not found at %s — DB-backed counts unavailable", path)
        return None
    try:
        import duckdb
    except ImportError:
        LOG.warning("duckdb not installed; DB-backed counts unavailable")
        return None
    try:
        return duckdb.connect(path, read_only=True)
    except Exception as exc:
        LOG.warning("Cannot open DuckDB at %s: %s", path, exc)
        return None


def _table_columns(con, table: str) -> set[str]:
    try:
        return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _table_stats(con, table: Optional[str], where: Optional[str], iso_col: Optional[str]):
    """(rows, countries) for a table/filter. Either may be None on failure/absence."""
    if con is None or not table:
        return (None, None)
    cols = _table_columns(con, table)
    if not cols:
        return (None, None)
    clause = f" WHERE {where}" if where else ""
    rows: Optional[int] = None
    countries: Optional[int] = None
    try:
        rows = con.execute(f"SELECT COUNT(*) FROM {table}{clause}").fetchone()[0]
    except Exception as exc:
        LOG.warning("row count failed for %s: %s", table, exc)
    if iso_col and iso_col.lower() in cols:
        try:
            countries = con.execute(
                f"SELECT COUNT(DISTINCT {iso_col}) FROM {table}{clause}"
            ).fetchone()[0]
        except Exception:
            countries = None
    return (rows, countries)


def _crisiswatch_edition(con) -> dict:
    """Latest CrisisWatch edition + freshness."""
    out: dict[str, Any] = {"state": "missing", "rows": 0, "edition": None, "months_old": None}
    if con is None:
        out["state"] = "unknown"
        return out
    if "crisiswatch_entries" not in {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name='crisiswatch_entries'"
    ).fetchall()}:
        return out
    row = con.execute(
        "SELECT COUNT(*), MAX(year * 100 + month) FROM crisiswatch_entries"
    ).fetchone()
    if not row or not row[0]:
        out["state"] = "empty"
        return out
    out["rows"] = row[0]
    ym = row[1]
    if ym:
        y, m = ym // 100, ym % 100
        out["edition"] = f"{y:04d}-{m:02d}"
        today = date.today()
        months_old = (today.year - y) * 12 + (today.month - m)
        out["months_old"] = months_old
        out["state"] = "current" if months_old <= 2 else "stale"
    return out


def _conflict_forecast_vintages(con) -> list[dict]:
    """Per-source latest forecast_issue_date + age, from conflict_forecasts."""
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT source, MAX(forecast_issue_date) FROM conflict_forecasts "
            "GROUP BY source ORDER BY source"
        ).fetchall()
    except Exception as exc:
        LOG.warning("conflict_forecasts vintage query failed: %s", exc)
        return []
    out: list[dict] = []
    today = date.today()
    for source, latest in rows:
        if latest is None:
            continue
        if hasattr(latest, "date") and not isinstance(latest, date):
            latest = latest.date()
        if isinstance(latest, str):
            try:
                latest = date.fromisoformat(latest[:10])
            except ValueError:
                continue
        age = (today - latest).days
        out.append({
            "source": str(source),
            "latest": latest.isoformat(),
            "age_days": age,
            "stale": age > _CF_STALENESS_WARN_DAYS,
        })
    return out


def _country_list_shrink(repo_root: Optional[Path] = None) -> Optional[dict]:
    """Detect a shrunken fewsnet_countries.json (on-disk vs git HEAD).

    The connector merge-union write should make shrinks impossible; this is
    the backstop that makes one visible if it ever happens again (a shrunken
    list silently drops DR question coverage and corrupts the IPC connector's
    FEWS NET exclusion set).
    """
    root = repo_root or _REPO_ROOT
    rel = "resolver/data/fewsnet_countries.json"
    path = root / rel
    if not path.exists():
        return None
    try:
        on_disk = len(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    in_git: Optional[int] = None
    try:
        import subprocess

        res = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            capture_output=True, text=True, cwd=str(root), timeout=10,
        )
        if res.returncode == 0:
            in_git = len(json.loads(res.stdout))
    except Exception:
        in_git = None
    return {
        "on_disk": on_disk,
        "in_git": in_git,
        "shrunk": in_git is not None and on_disk < in_git,
    }


def _attribution_drops(entries: list[dict]) -> dict:
    """Whole-country ACLED political attribution drops, from the JSONL extras."""
    for e in entries:
        if e.get("connector_id") == "acled_political_events":
            extras = e.get("extras") or {}
            countries = extras.get("attribution_dropped_countries") or []
            if countries:
                return {
                    "countries": [str(c) for c in countries],
                    "events": extras.get("attribution_dropped_events"),
                }
    return {}


def _ingest_windows_line() -> str:
    """Echo the effective month-window envs (unset = connector default)."""
    parts = []
    for var, default in (
        ("FEWSNET_MONTHS", "12"),
        ("IPC_API_MONTHS", "24"),
        ("GDACS_MONTHS", "3"),
    ):
        raw = os.environ.get(var)
        parts.append(f"{var}={raw}" if raw else f"{var}={default} (default)")
    return "Ingest windows: " + " · ".join(parts)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def build_phase_summary(entries: list[dict], con=None) -> str:
    if not entries:
        return (
            "⚠️ **No connector diagnostics found** "
            "(`connectors_report.jsonl` missing or empty). "
            "Check `DIAGNOSTICS_REPORT_PATH`.\n"
        )

    # Enrich each entry with phase + authoritative DB counts once.
    enriched: list[dict] = []
    for e in entries:
        cid = e.get("connector_id", "?")
        phase, table, where, iso_col = _CONNECTORS.get(cid, (0, None, None, None))
        rows_db, countries = _table_stats(con, table, where, iso_col)
        enriched.append({
            "cid": cid,
            "phase": phase,
            "status": e.get("status", "?"),
            "reason": e.get("reason"),
            "duration_ms": e.get("duration_ms", 0),
            "rows_run": _rows_written_this_run(e),
            "rows_db": rows_db,
            "countries": countries,
            "has_table": table is not None,
        })

    total = len(enriched)
    ok = sum(1 for e in enriched if e["status"] == "ok")
    errs = [e for e in enriched if e["status"] == "error"]
    skipped = [e for e in enriched if e["status"] == "skipped"]
    # "Succeeded but empty" — ok status yet the target table has 0 rows.
    empty_ok = [
        e for e in enriched
        if e["status"] == "ok" and e["has_table"] and e["rows_db"] == 0
    ]

    # Cross-cutting checks (feed the verdict + Problems & Warnings).
    cf_vintages = _conflict_forecast_vintages(con)
    stale_vintages = [v for v in cf_vintages if v["stale"]]
    shrink = _country_list_shrink()
    drops = _attribution_drops(entries)

    lines: list[str] = []
    lines.append("# \U0001f6f0️ Resolver Update — Run Summary\n")

    # --- Verdict -----------------------------------------------------------
    has_warnings = bool(
        empty_ok or stale_vintages or (shrink and shrink["shrunk"]) or drops
    )
    if errs:
        verdict = "❌"
    elif has_warnings:
        verdict = "⚠️"
    else:
        verdict = "✅"
    parts = [f"{ok}/{total} sources OK"]
    if errs:
        parts.append(f"{len(errs)} error{'s' if len(errs) != 1 else ''}")
    if skipped:
        parts.append(f"{len(skipped)} skipped")
    if empty_ok:
        parts.append(f"{len(empty_ok)} wrote 0 rows")
    if stale_vintages:
        parts.append(f"{len(stale_vintages)} stale forecast vintage{'s' if len(stale_vintages) != 1 else ''}")
    lines.append(f"**Result: {verdict} " + " · ".join(parts) + "**\n")

    # Per-phase roll-up
    rollup = []
    for pn in (1, 2, 3, 4):
        pe = [e for e in enriched if e["phase"] == pn]
        if pe:
            pok = sum(1 for e in pe if e["status"] == "ok")
            rollup.append(f"P{pn} {pok}/{len(pe)}")
    if rollup:
        lines.append("Phase roll-up: " + " · ".join(rollup) + "\n")

    # facts_resolved headline (what the resolution core holds after the run)
    if con is not None:
        fr_rows, fr_ctys = _table_stats(con, "facts_resolved", None, "iso3")
        if fr_rows is not None:
            lines.append(
                f"`facts_resolved`: **{_fmt_int(fr_rows)}** rows across "
                f"**{_fmt_int(fr_ctys)}** countries.\n"
            )

    lines.append(_ingest_windows_line() + "\n")

    # --- Problems & Warnings (surfaced first) ------------------------------
    lines.append("\n## Problems & Warnings\n")
    has_problems = bool(
        errs or skipped or empty_ok or stale_vintages
        or (shrink and shrink["shrunk"]) or drops
    )
    if not has_problems:
        lines.append("✅ None — every source ran and wrote data.\n")
    else:
        for e in errs:
            reason = e["reason"] or "no reason recorded"
            lines.append(f"- ❌ **{e['cid']}** (P{e['phase']}): {reason} — wrote 0 rows.")
        for e in empty_ok:
            lines.append(
                f"- ⚠️ **{e['cid']}** (P{e['phase']}): reported OK but "
                f"`{_CONNECTORS[e['cid']][1]}` is **empty** — likely a silent failure."
            )
        for v in stale_vintages:
            lines.append(
                f"- ⚠️ **{v['source']}** conflict forecast vintage is "
                f"**{v['age_days']} days old** (latest issue {v['latest']}, "
                f"> {_CF_STALENESS_WARN_DAYS}d) — prompts will carry a stale vintage; "
                f"if this persists across cycles, escalate upstream."
            )
        if shrink and shrink["shrunk"]:
            lines.append(
                f"- ⚠️ **fewsnet_countries.json SHRANK**: {shrink['on_disk']} entries "
                f"on disk vs {shrink['in_git']} in git — a short fetch window "
                f"overwrote the curated list (DR question coverage and the IPC "
                f"exclusion set are affected)."
            )
        if drops:
            n_ev = drops.get("events")
            ev_txt = f" ({_fmt_int(n_ev)} events discarded)" if isinstance(n_ev, int) else ""
            lines.append(
                f"- ⚠️ **acled_political_events**: {len(drops['countries'])} "
                f"countries returned events but attributed NONE — stored "
                f"nothing for: {', '.join(sorted(drops['countries']))}{ev_txt}. "
                f"Check countries.csv/alias name forms."
            )
        for e in skipped:
            reason = e["reason"] or "skipped"
            lines.append(f"- ⏭️ **{e['cid']}** (P{e['phase']}): {reason}.")
        lines.append("")

    # --- Per-phase detail tables (uniform columns) -------------------------
    by_phase: dict[int, list[dict]] = {}
    for e in enriched:
        by_phase.setdefault(e["phase"], []).append(e)

    for pn in sorted(k for k in by_phase if k):
        pe = by_phase[pn]
        # Drop the redundant idmc/skipped row when idmc_helix (or idmc) succeeded.
        if pn == 1 and any(e["cid"] in ("idmc", "idmc_helix") and e["status"] == "ok" for e in pe):
            pe = [e for e in pe if not (e["cid"] == "idmc" and e["status"] == "skipped")]
        lines.append(f"\n## {_PHASE_NAMES.get(pn, f'Phase {pn}')}\n")
        lines.append("| Source | Status | Rows in DB | Δ this run | Countries | Duration |")
        lines.append("|--------|:------:|-----------:|-----------:|----------:|---------:|")
        for e in sorted(pe, key=lambda x: x["cid"]):
            lines.append(
                f"| {e['cid']} | {_status_icon(e['status'])} | {_fmt_int(e['rows_db'])} "
                f"| {_fmt_int(e['rows_run'])} | {_fmt_int(e['countries'])} "
                f"| {_fmt_duration(e['duration_ms'])} |"
            )

    # Unknown-phase connectors (defensive — should be empty)
    unknown = by_phase.get(0)
    if unknown:
        lines.append("\n## Other (unmapped connectors)\n")
        lines.append("| Source | Status | Δ this run | Duration |")
        lines.append("|--------|:------:|-----------:|---------:|")
        for e in sorted(unknown, key=lambda x: x["cid"]):
            lines.append(
                f"| {e['cid']} | {_status_icon(e['status'])} "
                f"| {_fmt_int(e['rows_run'])} | {_fmt_duration(e['duration_ms'])} |"
            )

    # --- Conflict forecast vintages -----------------------------------------
    if cf_vintages:
        lines.append("\n## Conflict Forecast Vintages\n")
        lines.append("| Source | Latest issue | Age |")
        lines.append("|--------|--------------|----:|")
        for v in cf_vintages:
            flag = " ⚠️" if v["stale"] else ""
            lines.append(f"| {v['source']} | {v['latest']} | {v['age_days']}d{flag} |")

    # --- CrisisWatch edition ----------------------------------------------
    cw = _crisiswatch_edition(con)
    lines.append("\n## ICG CrisisWatch\n")
    if cw["state"] in ("current", "stale"):
        icon = "✅" if cw["state"] == "current" else "⚠️"
        lines.append(
            f"- {icon} Latest edition **{cw['edition']}** "
            f"({_fmt_int(cw['rows'])} rows, ~{cw['months_old']} month(s) old)."
        )
    elif cw["state"] == "empty":
        lines.append("- ⚠️ `crisiswatch_entries` is **empty** — check the CrisisWatch refresh.")
    elif cw["state"] == "missing":
        lines.append("- ❌ `crisiswatch_entries` table **not found**.")
    else:
        lines.append("- CrisisWatch status unavailable (DB not queryable).")

    lines.append(
        "\n_Rows in DB = authoritative count in the target table after this run. "
        "Δ this run = rows written this cycle (self-storing/direct sources only). "
        "Countries = distinct iso3 in the table._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Produce the consolidated Resolver Update run summary",
    )
    parser.add_argument(
        "--report",
        default="diagnostics/ingestion/connectors_report.jsonl",
        help="Path to the connectors_report.jsonl file",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to resolver.duckdb for authoritative counts (auto-detected if unset)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "github-step-summary"],
        default="markdown",
        help="Output format (both emit Markdown; retained for compatibility)",
    )
    args = parser.parse_args(argv)

    entries = _load_jsonl(args.report)
    con = _open_db(args.db)
    try:
        summary = build_phase_summary(entries, con=con)
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    print(summary)


if __name__ == "__main__":
    main()
