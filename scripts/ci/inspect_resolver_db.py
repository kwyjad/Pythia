# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver DuckDB — comprehensive inspection report generator.

Extracted from the heredoc formerly embedded in
``.github/workflows/inspect_resolver_duckdb.yml`` (July 2026) so the report
logic is testable and reviewable. The workflow now calls:

    python -m scripts.ci.inspect_resolver_db --db data/resolver.duckdb \
        --out resolver_inspect.md

Only stdlib + duckdb are required (the inspect workflow installs no pandas).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import duckdb

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTRY_LIST = _REPO_ROOT / "horizon_scanner" / "hs_country_list.txt"
DEFAULT_COUNTRIES_CSV = _REPO_ROOT / "resolver" / "data" / "countries.csv"


def build_report(
    db_path: Path,
    country_list_path: Optional[Path] = None,
    countries_csv_path: Optional[Path] = None,
) -> str:
    """Render the full inspection report for ``db_path`` as Markdown."""
    lines = []

    def L(text=""):
        lines.append(text)


    def safe_query(con, query, fallback="(query failed)"):
        try:
            return con.execute(query).fetchall()
        except Exception as e:
            L(f"_Error: {type(e).__name__}: {e}_")
            return []

    def table_exists(con, name):
        rows = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name=?",
            [name],
        ).fetchone()
        return rows and rows[0] > 0

    def row_count(con, name):
        if not table_exists(con, name):
            return None
        rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        return rows[0] if rows else 0

    def schema_info(con, name):
        if not table_exists(con, name):
            return []
        return con.execute(f"PRAGMA table_info('{name}')").fetchall()

    def fmt_num(n):
        if n is None:
            return "—"
        return f"{n:,}"

    def fmt_float(n, nd=4):
        # None-safe float formatter. Aggregate queries with no GROUP BY
        # return one all-NULL row on an empty table (e.g. AVG/SUM over
        # zero rows -> None); applying a numeric format spec to None
        # raises TypeError. Guard at the format site so a NULL in a
        # non-empty result is also handled.
        if n is None:
            return "—"
        return f"{n:,.{nd}f}"

    def _colset(con, table):
        # Lowercased column names for a table (PRAGMA table_info idx 1 = name).
        return {str(c[1]).lower() for c in schema_info(con, table)}

    def newest(con, table, candidates, chrono=False):
        # MAX() over the first candidate column that actually exists.
        # Returns (column, value) or (None, None) when nothing matches —
        # so a renamed/absent column degrades gracefully instead of raising.
        # chrono=True parses values chronologically instead of trusting SQL
        # MAX (needed for VARCHAR "%b%Y" labels, where MAX is alphabetical).
        if not table_exists(con, table):
            return None, None
        cols = _colset(con, table)
        for c in candidates:
            if c.lower() in cols:
                try:
                    if chrono:
                        _mn, _mx = chrono_min_max(con, table, c)
                        if _mx is not None:
                            return c, _mx
                        continue
                    v = con.execute(f"SELECT MAX({c}) FROM {table}").fetchone()
                    return c, (v[0] if v else None)
                except Exception:
                    continue
        return None, None

    def _age_days(val, now):
        # Best-effort age in days for YYYY-MM / YYYY-MM-DD / timestamp /
        # MonYYYY ("Jul2026", the ACAPS snapshot label) values (str, date, or
        # datetime). None when unparseable (→ no staleness flag).
        if val is None:
            return None
        s = str(val).strip()
        for txt, fmt in (
            (s[:19], "%Y-%m-%d %H:%M:%S"),
            (s[:19], "%Y-%m-%dT%H:%M:%S"),
            (s[:10], "%Y-%m-%d"),
            (s[:7], "%Y-%m"),
            (s, "%b%Y"),
        ):
            try:
                return (now - datetime.strptime(txt, fmt)).days
            except ValueError:
                continue
        return None

    def _parse_flex_date(val):
        # Chronological parse for mixed date labels (ISO forms + the ACAPS
        # "%b%Y" snapshot label). Returns a datetime or None.
        if val is None:
            return None
        s = str(val).strip()
        for txt, fmt in (
            (s[:19], "%Y-%m-%d %H:%M:%S"),
            (s[:19], "%Y-%m-%dT%H:%M:%S"),
            (s[:10], "%Y-%m-%d"),
            (s[:7], "%Y-%m"),
            (s, "%b%Y"),
        ):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue
        return None

    def chrono_min_max(con, table, col):
        # Chronological MIN/MAX for a VARCHAR date-label column. A plain SQL
        # MIN/MAX sorts "%b%Y" labels alphabetically ("Apr2026" < "Sep2025"),
        # which produced nonsense ranges in earlier reports.
        vals = safe_query(con, f"SELECT DISTINCT {col} FROM {table}")
        parsed = [(v[0], _parse_flex_date(v[0])) for v in vals]
        parsed = [(raw, dt) for raw, dt in parsed if dt is not None]
        if not parsed:
            return None, None
        parsed.sort(key=lambda x: x[1])
        return parsed[0][0], parsed[-1][0]

    def _fresh_status(rows, newest_val, stale_days, now):
        if not rows:
            return "❌ EMPTY"
        age = _age_days(newest_val, now)
        if age is None:
            return "—"
        if age < 0:
            # Future-dated (e.g. a projection/forecast horizon reaching
            # ahead, like FEWS NET phase3plus_projection) — never stale.
            return "✅ current (future-dated)"
        if age > stale_days:
            return f"⚠️ {age}d old"
        return f"✅ {age}d"

    # =====================================================================
    if not db_path.exists():
        L("# Resolver DuckDB Inspection")
        L()
        L(f"_Database not found at `{db_path}`_")
        return "\n".join(lines)

    print(f"[inspect] Connecting to {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)

    db_size = db_path.stat().st_size
    db_size_mb = db_size / (1024 * 1024)

    L("# Resolver DuckDB — Comprehensive Inspection Report")
    L()
    L(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_")
    L(f"_Database size: {db_size_mb:.1f} MB ({db_size:,} bytes)_")
    L()

    # =====================================================================
    # DATA FRESHNESS — AT A GLANCE
    # The single "never have to guess" view: newest data point + row count
    # per source, with a per-source staleness flag. Detailed per-table
    # sections follow below.
    # =====================================================================
    L("## Data Freshness — At a Glance")
    L()
    L("_Newest data point and row count per source, with a staleness flag "
      "computed against this report's generation time. ✅ current · "
      "⚠️ past its refresh cadence · ❌ empty · — undated/unparseable. "
      "The parenthesised name is the column the 'Newest' value came from. "
      "Thresholds are per-source refresh cadences, not hard SLAs._")
    L()
    L("| Source | Rows | Newest | Status |")
    L("|--------|-----:|--------|--------|")

    _now = datetime.utcnow()

    # facts_resolved: split observations from forward-looking projections —
    # a projection ym reaching months ahead (e.g. FEWS NET
    # phase3plus_projection to 2027-02) used to mask the newest actual
    # observation date behind a "future-dated" flag.
    if table_exists(con, "facts_resolved"):
        _obs = safe_query(con, """
            SELECT COUNT(*), MAX(ym) FROM facts_resolved
            WHERE lower(metric) <> 'phase3plus_projection'
        """)
        if _obs and _obs[0][0]:
            _on, _oym = _obs[0]
            L(f"| Resolution facts · observations | {fmt_num(_on)} | {_oym} (ym) | "
              f"{_fresh_status(_on, _oym, 45, _now)} |")
        _proj = safe_query(con, """
            SELECT COUNT(*), MAX(ym) FROM facts_resolved
            WHERE lower(metric) = 'phase3plus_projection'
        """)
        if _proj and _proj[0][0]:
            _pn, _pym = _proj[0]
            L(f"| Resolution facts · projections (phase3plus_projection) | "
              f"{fmt_num(_pn)} | {_pym} (ym) | {_fresh_status(_pn, _pym, 45, _now)} |")
    else:
        L("| Resolution facts (facts_resolved) | _n/a_ | (no table) | — |")

    # (label, table, [candidate date/label columns, newest-first], stale_days,
    #  chrono) — chrono=True for VARCHAR "%b%Y" label columns where SQL MAX
    #  sorts alphabetically.
    _freshness_sources = [
        ("ACLED monthly fatalities", "acled_monthly_fatalities", ["month", "ym"], 45, False),
        ("ACLED political events", "acled_political_events", ["event_date", "fetched_at"], 45, False),
        ("GDELT conflict indicators", "gdelt_conflict_indicators", ["event_date", "fetched_at"], 45, False),
        ("NMME seasonal forecasts", "seasonal_forecasts", ["forecast_issue_date"], 45, False),
        ("ReliefWeb reports", "reliefweb_reports", ["published_date", "created_at", "fetched_at"], 45, False),
        ("ACAPS INFORM severity", "acaps_inform_severity", ["snapshot_date", "fetched_at"], 120, True),
        ("ACAPS daily monitoring", "acaps_daily_monitoring", ["entry_date", "fetched_at"], 45, False),
        ("ACAPS risk radar", "acaps_risk_radar", ["fetched_at"], 120, False),
        ("ACAPS humanitarian access", "acaps_humanitarian_access", ["snapshot_date", "fetched_at"], 120, True),
        ("HDX Signals", "hdx_signals", ["signal_date", "fetched_at"], 60, False),
        ("ENSO state", "enso_state", ["fetch_date"], 45, False),
        ("Seasonal TC outlooks", "seasonal_tc_outlooks", ["fetched_at"], 120, False),
    ]
    for _label, _table, _cands, _stale, _chrono in _freshness_sources:
        _n = row_count(con, _table)
        if _n is None:
            L(f"| {_label} | _n/a_ | (no table) | — |")
            continue
        _col, _val = newest(con, _table, _cands, chrono=_chrono)
        _status = _fresh_status(_n, _val, _stale, _now)
        _shown = f"{_val} ({_col})" if _val is not None else "—"
        L(f"| {_label} | {fmt_num(_n)} | {_shown} | {_status} |")

    # conflict_forecasts — one row per source; CAST staleness hides behind an
    # overall MAX (a fresh conflictforecast.org vintage would mask a stale CAST).
    if table_exists(con, "conflict_forecasts") and row_count(con, "conflict_forecasts"):
        _cf = safe_query(con, """
            SELECT source, COUNT(*) AS n, MAX(forecast_issue_date) AS newest
            FROM conflict_forecasts GROUP BY source ORDER BY source
        """)
        for _src, _n, _newest in _cf:
            L(f"| Conflict forecast · {_src} | {fmt_num(_n)} | {_newest} | "
              f"{_fresh_status(_n, _newest, 45, _now)} |")

    # CrisisWatch — the edition month we most often have to guess.
    if table_exists(con, "crisiswatch_entries"):
        _cwn = row_count(con, "crisiswatch_entries")
        _ed = safe_query(con, """
            SELECT year, month, COUNT(*) AS n
            FROM crisiswatch_entries GROUP BY year, month
            ORDER BY year DESC, month DESC LIMIT 1
        """) if _cwn else []
        if _ed:
            _y, _m, _cnt = _ed[0]
            _ym = f"{int(_y):04d}-{int(_m):02d}"
            L(f"| CrisisWatch (latest edition) | {fmt_num(_cwn)} | "
              f"{_ym} ({_cnt} entries) | {_fresh_status(_cwn, _ym, 60, _now)} |")
        else:
            L(f"| CrisisWatch (latest edition) | {fmt_num(_cwn or 0)} | — | ❌ EMPTY |")
    L()

    # =====================================================================
    # PIPELINE STAGE COMPLETENESS
    # At a glance: is this a resolver-only DB, or a full end-to-end run?
    # (An easy-to-miss "resolutions/scores = 0" means scoring hasn't run.)
    # =====================================================================
    L("## Pipeline Stage Completeness")
    L()
    L("_Row presence per pipeline stage, in run order. Tells you at a glance "
      "whether this DB is resolver-only or a full end-to-end run. "
      "✓ populated · ✗ empty (stage not run / no output)._")
    L()
    L("| # | Stage | Table | Rows | |")
    L("|--:|-------|-------|-----:|:-:|")
    _stages = [
        ("Resolver facts", "facts_resolved"),
        ("HS runs", "hs_runs"),
        ("HS triage", "hs_triage"),
        ("Questions", "questions"),
        ("Forecasts (raw)", "forecasts_raw"),
        ("Forecasts (ensemble)", "forecasts_ensemble"),
        ("Sibyl forecasts", "sibyl_forecasts"),
        ("Resolutions (ground truth)", "resolutions"),
        ("Scores", "scores"),
        ("Calibration weights", "calibration_weights"),
        ("Calibration advice", "calibration_advice"),
        ("Source coverage map", "source_coverage"),
    ]
    for _i, (_lbl, _t) in enumerate(_stages, 1):
        _rc = row_count(con, _t)
        _mark = "✓" if _rc else "✗"
        _rcs = fmt_num(_rc) if _rc is not None else "_(no table)_"
        L(f"| {_i} | {_lbl} | `{_t}` | {_rcs} | {_mark} |")
    L()

    # =====================================================================
    # INJECT READINESS — TARGET-COUNTRY COVERAGE PER SOURCE
    # Directly answers "are the injects ready for a full target-list run?"
    # =====================================================================
    def _load_target_iso3s():
        """Resolve the HS target country list to ISO3 codes via countries.csv."""
        clp = Path(country_list_path) if country_list_path else DEFAULT_COUNTRY_LIST
        ccp = Path(countries_csv_path) if countries_csv_path else DEFAULT_COUNTRIES_CSV
        if not clp.exists() or not ccp.exists():
            return [], []
        name_to_iso3 = {}
        iso3s_known = set()
        with open(ccp, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                iso3 = (row.get("iso3") or "").strip().upper()
                name = (row.get("country_name") or "").strip()
                if iso3 and name:
                    name_to_iso3[name.lower()] = iso3
                    iso3s_known.add(iso3)
        targets, unresolved = [], []
        for raw in clp.read_text(encoding="utf-8").splitlines():
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                continue
            up = entry.upper()
            if len(up) == 3 and up.isalpha() and up in iso3s_known:
                targets.append(up)
            elif entry.lower() in name_to_iso3:
                targets.append(name_to_iso3[entry.lower()])
            else:
                unresolved.append(entry)
        seen = set()
        targets = [t for t in targets if not (t in seen or seen.add(t))]
        return targets, unresolved

    _targets, _unresolved_targets = _load_target_iso3s()
    if _targets:
        _tset = set(_targets)
        L(f"## Inject Readiness — {len(_targets)}-Country Target List")
        L()
        L("_Per-source coverage of the HS target list "
          "(`horizon_scanner/hs_country_list.txt`): how many target countries "
          "have at least one row in each prompt-inject source. 'Expectation' "
          "calibrates the reading — global sources should cover ~all targets; "
          "crisis- or hazard-scoped sources are partial by design. Expand the "
          "details block to see which target countries are missing._")
        L()
        if _unresolved_targets:
            L(f"> **WARNING:** {len(_unresolved_targets)} target-list entries did "
              f"not resolve to ISO3: {', '.join(_unresolved_targets)}")
            L()

        # (label, table for existence check, distinct-iso3 SQL, expectation)
        _inject_sources = [
            ("ACLED monthly fatalities", "acled_monthly_fatalities",
             "SELECT DISTINCT iso3 FROM acled_monthly_fatalities", "~all"),
            ("GDACS event history (facts_resolved event_occurrence)", "facts_resolved",
             "SELECT DISTINCT iso3 FROM facts_resolved WHERE lower(metric)='event_occurrence'",
             "most (only countries with recorded events)"),
            ("Food security Phase 3+ (FEWS NET/IPC)", "facts_resolved",
             "SELECT DISTINCT iso3 FROM facts_resolved WHERE lower(metric)='phase3plus_in_need'",
             "partial (monitored countries only)"),
            ("Conflict forecast · VIEWS", "conflict_forecasts",
             "SELECT DISTINCT iso3 FROM conflict_forecasts WHERE source='VIEWS'", "~all"),
            ("Conflict forecast · conflictforecast.org", "conflict_forecasts",
             "SELECT DISTINCT iso3 FROM conflict_forecasts WHERE source='conflictforecast_org'", "~all"),
            ("Conflict forecast · ACLED CAST", "conflict_forecasts",
             "SELECT DISTINCT iso3 FROM conflict_forecasts WHERE source='ACLED_CAST'",
             "partial (CAST coverage)"),
            ("NMME seasonal forecasts", "seasonal_forecasts",
             "SELECT DISTINCT iso3 FROM seasonal_forecasts", "~all"),
            ("ReliefWeb reports", "reliefweb_reports",
             "SELECT DISTINCT iso3 FROM reliefweb_reports",
             "most (report-generating situations)"),
            ("ACLED political events", "acled_political_events",
             "SELECT DISTINCT iso3 FROM acled_political_events",
             "most (countries with recent events)"),
            ("GDELT conflict indicators", "gdelt_conflict_indicators",
             "SELECT DISTINCT iso3 FROM gdelt_conflict_indicators",
             "most (media-covered countries)"),
            ("ACAPS INFORM severity", "acaps_inform_severity",
             "SELECT DISTINCT iso3 FROM acaps_inform_severity",
             "partial (crisis countries only)"),
            ("HDX Signals", "hdx_signals",
             "SELECT DISTINCT iso3 FROM hdx_signals",
             "partial (signal-emitting countries)"),
            ("Seasonal TC context cache", "seasonal_tc_context_cache",
             "SELECT DISTINCT iso3 FROM seasonal_tc_context_cache",
             "partial (TC-exposed countries only)"),
            ("CrisisWatch (latest edition)", "crisiswatch_entries",
             """SELECT DISTINCT iso3 FROM crisiswatch_entries
                WHERE year * 100 + month = (SELECT MAX(year * 100 + month)
                                            FROM crisiswatch_entries)""",
             "partial (ICG-monitored countries)"),
        ]

        _missing_details = []
        L("| Inject source | Target coverage | Missing | Expectation |")
        L("|---------------|----------------:|--------:|-------------|")
        for _lbl, _tbl, _q, _expect in _inject_sources:
            if not table_exists(con, _tbl):
                L(f"| {_lbl} | _no table_ | — | {_expect} |")
                continue
            _cov = {str(r[0]).strip().upper() for r in safe_query(con, _q) if r and r[0]}
            _hit = _tset & _cov
            _miss = sorted(_tset - _cov)
            L(f"| {_lbl} | {len(_hit)}/{len(_targets)} | {len(_miss)} | {_expect} |")
            if _miss:
                _missing_details.append((_lbl, _miss))
        L()
        if _missing_details:
            L("<details>")
            L("<summary>Missing target countries per source</summary>")
            L()
            for _lbl, _miss in _missing_details:
                _shown = ", ".join(_miss[:60])
                if len(_miss) > 60:
                    _shown += f", … (+{len(_miss) - 60} more)"
                L(f"- **{_lbl}** ({len(_miss)}): {_shown}")
            L()
            L("</details>")
            L()

    # =====================================================================
    # 1. TABLE INVENTORY
    # =====================================================================
    L("## 1. Table Inventory")
    L()

    all_tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()

    L(f"**Total tables: {len(all_tables)}**")
    L()
    L("| Table | Rows | Columns |")
    L("|-------|-----:|--------:|")
    for (tbl,) in all_tables:
        n = row_count(con, tbl)
        cols = len(schema_info(con, tbl))
        L(f"| `{tbl}` | {fmt_num(n)} | {cols} |")
    L()

    # =====================================================================
    # 2. RESOLVER CORE TABLES
    # =====================================================================
    L("## 2. Resolver Core Tables")
    L()

    # --- facts_resolved ---
    L("### facts_resolved")
    if table_exists(con, "facts_resolved"):
        n = row_count(con, "facts_resolved")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(hazard_code, '(null)') AS hazard_code,
                COALESCE(metric, '(null)') AS metric,
                COUNT(*) AS n,
                COUNT(DISTINCT iso3) AS n_countries,
                MIN(ym) AS min_ym,
                MAX(ym) AS max_ym
            FROM facts_resolved
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("| hazard_code | metric | rows | countries | min_ym | max_ym |")
            L("|-------------|--------|-----:|----------:|--------|--------|")
            for hz, m, n, nc, mn, mx in rows:
                L(f"| {hz} | {m} | {fmt_num(n)} | {nc} | {mn} | {mx} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- facts_deltas ---
    L("### facts_deltas")
    if table_exists(con, "facts_deltas"):
        n = row_count(con, "facts_deltas")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(hazard_code, '(null)') AS hazard_code,
                COALESCE(metric, '(null)') AS metric,
                COUNT(*) AS n,
                COUNT(DISTINCT iso3) AS n_countries,
                MIN(ym) AS min_ym,
                MAX(ym) AS max_ym
            FROM facts_deltas
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("| hazard_code | metric | rows | countries | min_ym | max_ym |")
            L("|-------------|--------|-----:|----------:|--------|--------|")
            for hz, m, n, nc, mn, mx in rows:
                L(f"| {hz} | {m} | {fmt_num(n)} | {nc} | {mn} | {mx} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- emdat_pa ---
    L("### emdat_pa")
    if table_exists(con, "emdat_pa"):
        n = row_count(con, "emdat_pa")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(shock_type, '(null)') AS shock_type,
                COUNT(*) AS n,
                COUNT(DISTINCT iso3) AS n_countries,
                MIN(ym) AS min_ym,
                MAX(ym) AS max_ym
            FROM emdat_pa
            GROUP BY 1
            ORDER BY 1
        """)
        if rows:
            L("| shock_type | rows | countries | min_ym | max_ym |")
            L("|------------|-----:|----------:|--------|--------|")
            for st, n, nc, mn, mx in rows:
                L(f"| {st} | {fmt_num(n)} | {nc} | {mn} | {mx} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- acled_monthly_fatalities ---
    L("### acled_monthly_fatalities")
    if table_exists(con, "acled_monthly_fatalities"):
        n = row_count(con, "acled_monthly_fatalities")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COUNT(DISTINCT iso3) AS n_countries,
                MIN(month) AS min_month,
                MAX(month) AS max_month,
                SUM(fatalities) AS total_fatalities
            FROM acled_monthly_fatalities
        """)
        if rows and rows[0]:
            nc, mn, mx, tf = rows[0]
            L(f"- Countries: {nc}")
            L(f"- Month range: {mn} → {mx}")
            L(f"- Total fatalities: {fmt_num(tf)}")
            L()

        # Top 10 countries by fatalities
        rows = safe_query(con, """
            SELECT iso3, SUM(fatalities) AS total
            FROM acled_monthly_fatalities
            GROUP BY iso3
            ORDER BY total DESC
            LIMIT 10
        """)
        if rows:
            L("Top 10 countries by total fatalities:")
            L()
            L("| iso3 | total_fatalities |")
            L("|------|----------------:|")
            for iso3, total in rows:
                L(f"| {iso3} | {fmt_num(total)} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # =====================================================================
    # 3. HORIZON SCANNER TABLES
    # =====================================================================
    L("## 3. Horizon Scanner Tables")
    L()

    # --- hs_runs ---
    L("### hs_runs")
    if table_exists(con, "hs_runs"):
        n = row_count(con, "hs_runs")
        L(f"Total runs: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT * FROM hs_runs
            ORDER BY generated_at DESC NULLS LAST
            LIMIT 5
        """)
        if rows:
            cols = [d[0] for d in con.description]
            L("Recent runs (up to 5):")
            L("```")
            for row in rows:
                L(str(dict(zip(cols, row))))
            L("```")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- hs_triage ---
    L("### hs_triage")
    if table_exists(con, "hs_triage"):
        n = row_count(con, "hs_triage")
        L(f"Total rows: **{fmt_num(n)}**")
        L()

        # By run_id
        rows = safe_query(con, """
            SELECT run_id, COUNT(*) AS n, COUNT(DISTINCT iso3) AS n_countries
            FROM hs_triage
            GROUP BY run_id
            ORDER BY run_id DESC
            LIMIT 5
        """)
        if rows:
            L("| run_id | rows | countries |")
            L("|--------|-----:|----------:|")
            for rid, n, nc in rows:
                L(f"| {rid} | {fmt_num(n)} | {nc} |")
            L()

        # Tier distribution (latest run)
        rows = safe_query(con, """
            SELECT tier, hazard_code, COUNT(*) AS n
            FROM hs_triage
            WHERE run_id = (SELECT run_id FROM hs_triage ORDER BY created_at DESC LIMIT 1)
            GROUP BY tier, hazard_code
            ORDER BY tier, hazard_code
        """)
        if rows:
            L("**Tier distribution (latest run):**")
            L()
            L("| tier | hazard_code | count |")
            L("|------|-------------|------:|")
            for tier, hz, n in rows:
                L(f"| {tier} | {hz} | {n} |")
            L()

        # RC level distribution (latest run)
        rows = safe_query(con, """
            SELECT
                regime_change_level,
                COUNT(*) AS n,
                ROUND(AVG(regime_change_likelihood), 3) AS avg_lk,
                ROUND(AVG(regime_change_magnitude), 3) AS avg_mag
            FROM hs_triage
            WHERE run_id = (SELECT run_id FROM hs_triage ORDER BY created_at DESC LIMIT 1)
            GROUP BY regime_change_level
            ORDER BY regime_change_level
        """)
        if rows:
            L("**Regime change level distribution (latest run):**")
            L()
            L("| RC level | count | avg_likelihood | avg_magnitude |")
            L("|:--------:|------:|---------------:|--------------:|")
            for lev, n, avg_lk, avg_mag in rows:
                L(f"| {lev} | {n} | {avg_lk} | {avg_mag} |")
            L()

        # Triage score distribution
        rows = safe_query(con, """
            SELECT
                CASE
                    WHEN triage_score < 0.25 THEN '0.00-0.24'
                    WHEN triage_score < 0.50 THEN '0.25-0.49'
                    WHEN triage_score < 0.75 THEN '0.50-0.74'
                    ELSE '0.75-1.00'
                END AS score_band,
                COUNT(*) AS n
            FROM hs_triage
            WHERE run_id = (SELECT run_id FROM hs_triage ORDER BY created_at DESC LIMIT 1)
            GROUP BY 1
            ORDER BY 1
        """)
        if rows:
            L("**Triage score distribution (latest run):**")
            L()
            L("| score_band | count |")
            L("|------------|------:|")
            for band, n in rows:
                L(f"| {band} | {n} |")
            L()

        # Track distribution
        rows = safe_query(con, """
            SELECT COALESCE(CAST(track AS TEXT), 'null') AS track, COUNT(*) AS n
            FROM hs_triage
            WHERE run_id = (SELECT run_id FROM hs_triage ORDER BY created_at DESC LIMIT 1)
            GROUP BY 1
            ORDER BY 1
        """)
        if rows:
            L("**Track distribution (latest run):**")
            L()
            L("| track | count |")
            L("|-------|------:|")
            for tr, n in rows:
                L(f"| {tr} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- hs_hazard_tail_packs ---
    L("### hs_hazard_tail_packs")
    if table_exists(con, "hs_hazard_tail_packs"):
        n = row_count(con, "hs_hazard_tail_packs")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT hazard_code, rc_level, COUNT(*) AS n
            FROM hs_hazard_tail_packs
            GROUP BY hazard_code, rc_level
            ORDER BY hazard_code, rc_level
        """)
        if rows:
            L()
            L("| hazard_code | rc_level | count |")
            L("|-------------|:--------:|------:|")
            for hz, lev, n in rows:
                L(f"| {hz} | {lev} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- hs_adversarial_checks ---
    L("### hs_adversarial_checks")
    if table_exists(con, "hs_adversarial_checks"):
        n = row_count(con, "hs_adversarial_checks")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT net_assessment, COUNT(*) AS n
            FROM hs_adversarial_checks
            GROUP BY net_assessment
            ORDER BY net_assessment
        """)
        if rows:
            L()
            L("| net_assessment | count |")
            L("|----------------|------:|")
            for na, n in rows:
                L(f"| {na} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- hs_country_reports ---
    L("### hs_country_reports")
    if table_exists(con, "hs_country_reports"):
        n = row_count(con, "hs_country_reports")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
    else:
        L("_Table does not exist._")
    L()

    # --- hs_scenarios ---
    L("### hs_scenarios")
    if table_exists(con, "hs_scenarios"):
        n = row_count(con, "hs_scenarios")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
    else:
        L("_Table does not exist._")
    L()

    # =====================================================================
    # 4. QUESTIONS & FORECASTS
    # =====================================================================
    L("## 4. Questions & Forecasts")
    L()

    # --- questions ---
    L("### questions")
    if table_exists(con, "questions"):
        n = row_count(con, "questions")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(status, '(null)') AS status,
                COALESCE(hazard_code, '(null)') AS hazard_code,
                COALESCE(metric, '(null)') AS metric,
                COUNT(*) AS n,
                COUNT(DISTINCT iso3) AS n_countries
            FROM questions
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
        """)
        if rows:
            L("| status | hazard_code | metric | questions | countries |")
            L("|--------|-------------|--------|----------:|----------:|")
            for st, hz, m, n, nc in rows:
                L(f"| {st} | {hz} | {m} | {n} | {nc} |")
            L()

        # Track distribution
        rows = safe_query(con, """
            SELECT COALESCE(CAST(track AS TEXT), 'null') AS track,
                   COUNT(*) AS n
            FROM questions
            WHERE status = 'active'
            GROUP BY 1 ORDER BY 1
        """)
        if rows:
            L("**Active questions by track:**")
            L()
            L("| track | count |")
            L("|-------|------:|")
            for tr, n in rows:
                L(f"| {tr} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- question_research ---
    L("### question_research")
    if table_exists(con, "question_research"):
        n = row_count(con, "question_research")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT run_id, COUNT(*) AS n
            FROM question_research
            GROUP BY run_id
            ORDER BY run_id DESC
            LIMIT 5
        """)
        if rows:
            L("| run_id | rows |")
            L("|--------|-----:|")
            for rid, n in rows:
                L(f"| {rid} | {fmt_num(n)} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- forecasts_raw ---
    L("### forecasts_raw")
    if table_exists(con, "forecasts_raw"):
        n = row_count(con, "forecasts_raw")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(model_name, '(null)') AS model,
                COALESCE(status, '(null)') AS status,
                COUNT(*) AS n
            FROM forecasts_raw
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("| model | status | rows |")
            L("|-------|--------|-----:|")
            for m, st, n in rows:
                L(f"| {m} | {st} | {fmt_num(n)} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- forecasts_ensemble ---
    L("### forecasts_ensemble")
    if table_exists(con, "forecasts_ensemble"):
        n = row_count(con, "forecasts_ensemble")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        rows = safe_query(con, """
            SELECT
                COALESCE(model_name, '(null)') AS model_name,
                COALESCE(status, '(null)') AS status,
                COUNT(DISTINCT question_id) AS n_questions,
                COUNT(*) AS n_rows
            FROM forecasts_ensemble
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("| model_name | status | questions | rows |")
            L("|------------|--------|----------:|-----:|")
            for mn, st, nq, nr in rows:
                L(f"| {mn} | {st} | {nq} | {fmt_num(nr)} |")
            L()

        # By hazard_code
        rows = safe_query(con, """
            SELECT
                COALESCE(hazard_code, '(null)') AS hazard_code,
                COALESCE(metric, '(null)') AS metric,
                COUNT(DISTINCT question_id) AS n_questions,
                COUNT(*) AS n_rows
            FROM forecasts_ensemble
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("By hazard/metric:")
            L()
            L("| hazard_code | metric | questions | rows |")
            L("|-------------|--------|----------:|-----:|")
            for hz, m, nq, nr in rows:
                L(f"| {hz} | {m} | {nq} | {fmt_num(nr)} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- scenarios ---
    L("### scenarios")
    if table_exists(con, "scenarios"):
        n = row_count(con, "scenarios")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT scenario_type, COUNT(*) AS n
            FROM scenarios
            GROUP BY scenario_type
            ORDER BY scenario_type
        """)
        if rows:
            L()
            L("| scenario_type | count |")
            L("|---------------|------:|")
            for st, n in rows:
                L(f"| {st} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # =====================================================================
    # 5. SCORING & CALIBRATION
    # =====================================================================
    L("## 5. Scoring & Calibration")
    L()

    # --- resolutions ---
    L("### resolutions")
    if table_exists(con, "resolutions"):
        n = row_count(con, "resolutions")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT COUNT(DISTINCT question_id) AS n_questions,
                   MIN(observed_month) AS min_month,
                   MAX(observed_month) AS max_month
            FROM resolutions
        """)
        if rows and rows[0]:
            nq, mn, mx = rows[0]
            L(f"- Resolved questions: {nq}")
            L(f"- Month range: {mn} → {mx}")
        L()
    else:
        L("_Table does not exist._")
    L()

    # --- scores ---
    L("### scores")
    if table_exists(con, "scores"):
        n = row_count(con, "scores")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT
                COALESCE(score_type, '(null)') AS score_type,
                COALESCE(model_name, 'ensemble') AS model,
                COUNT(*) AS n,
                ROUND(AVG(value), 4) AS avg_score
            FROM scores
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L()
            L("| score_type | model | rows | avg_score |")
            L("|------------|-------|-----:|----------:|")
            for st, m, n, avg in rows:
                L(f"| {st} | {m} | {fmt_num(n)} | {avg} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- calibration_weights ---
    L("### calibration_weights")
    if table_exists(con, "calibration_weights"):
        n = row_count(con, "calibration_weights")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT as_of_month, hazard_code, metric, model_name,
                   ROUND(weight, 4) AS weight, n_questions, n_samples,
                   ROUND(avg_brier, 4) AS avg_brier
            FROM calibration_weights
            ORDER BY as_of_month DESC, hazard_code, metric, model_name
            LIMIT 20
        """)
        if rows:
            L()
            L("| as_of_month | hazard | metric | model | weight | n_q | n_s | avg_brier |")
            L("|-------------|--------|--------|-------|-------:|----:|----:|----------:|")
            for am, hz, m, mn, w, nq, ns, ab in rows:
                L(f"| {am} | {hz} | {m} | {mn} | {w} | {nq} | {ns} | {ab} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- calibration_advice ---
    L("### calibration_advice")
    if table_exists(con, "calibration_advice"):
        n = row_count(con, "calibration_advice")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT as_of_month, hazard_code, metric,
                   LENGTH(advice) AS advice_len
            FROM calibration_advice
            ORDER BY as_of_month DESC, hazard_code, metric
            LIMIT 15
        """)
        if rows:
            L()
            L("| as_of_month | hazard_code | metric | advice_length |")
            L("|-------------|-------------|--------|-------------:|")
            for am, hz, m, al in rows:
                L(f"| {am} | {hz} | {m} | {fmt_num(al)} chars |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- bucket_definitions ---
    L("### bucket_definitions")
    if table_exists(con, "bucket_definitions"):
        rows = safe_query(con, "SELECT * FROM bucket_definitions ORDER BY metric, bucket_index")
        if rows:
            L(f"Total rows: **{len(rows)}**")
            L()
            L("| metric | bucket_index | label | lower_bound | upper_bound |")
            L("|--------|:------------:|-------|------------:|------------:|")
            for row in rows:
                # Schema: metric, bucket_index, label, lower_bound, upper_bound
                m, bi, lbl, lb, ub = row[0], row[1], row[2], row[3], row[4]
                L(f"| {m} | {bi} | {lbl} | {lb} | {ub} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- bucket_centroids ---
    L("### bucket_centroids")
    if table_exists(con, "bucket_centroids"):
        rows = safe_query(con, """
            SELECT hazard_code, metric, bucket_index, centroid
            FROM bucket_centroids
            ORDER BY hazard_code, metric, bucket_index
        """)
        if rows:
            L(f"Total rows: **{len(rows)}**")
            L()
            L("| hazard_code | metric | bucket_index | centroid |")
            L("|-------------|--------|:------------:|---------:|")
            for hz, m, bi, c in rows:
                L(f"| {hz} | {m} | {bi} | {fmt_float(c, 0)} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # =====================================================================
    # 6. LLM CALLS & TELEMETRY
    # =====================================================================
    L("## 6. LLM Calls & Telemetry")
    L()

    # --- llm_calls ---
    L("### llm_calls")
    if table_exists(con, "llm_calls"):
        n = row_count(con, "llm_calls")
        L(f"Total rows: **{fmt_num(n)}**")
        L()

        # By phase
        rows = safe_query(con, """
            SELECT
                COALESCE(phase, call_type) AS phase,
                COALESCE(status, '(null)') AS status,
                COUNT(*) AS n,
                ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS total_cost,
                SUM(COALESCE(total_tokens, 0)) AS total_tokens
            FROM llm_calls
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        if rows:
            L("| phase | status | calls | cost_usd | tokens |")
            L("|-------|--------|------:|---------:|-------:|")
            for ph, st, n, cost, tok in rows:
                L(f"| {ph} | {st} | {fmt_num(n)} | ${fmt_float(cost, 4)} | {fmt_num(tok)} |")
            L()

        # By provider/model
        rows = safe_query(con, """
            SELECT
                COALESCE(provider, '(null)') AS provider,
                COALESCE(model_id, model_name, '(null)') AS model,
                COUNT(*) AS n,
                ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS total_cost,
                ROUND(AVG(COALESCE(elapsed_ms, 0)), 0) AS avg_ms
            FROM llm_calls
            GROUP BY 1, 2
            ORDER BY total_cost DESC
            LIMIT 15
        """)
        if rows:
            L("**Cost by provider/model (top 15):**")
            L()
            L("| provider | model | calls | cost_usd | avg_ms |")
            L("|----------|-------|------:|---------:|-------:|")
            for prov, mod, n, cost, avg_ms in rows:
                L(f"| {prov} | {mod} | {fmt_num(n)} | ${fmt_float(cost, 4)} | {fmt_float(avg_ms, 0)} |")
            L()

        # Error summary
        rows = safe_query(con, """
            SELECT
                COALESCE(error_type, 'none') AS error_type,
                COUNT(*) AS n
            FROM llm_calls
            WHERE status != 'ok' OR error_type IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 10
        """)
        if rows:
            L("**Error summary:**")
            L()
            L("| error_type | count |")
            L("|------------|------:|")
            for et, n in rows:
                L(f"| {et} | {n} |")
            L()
    else:
        L("_Table does not exist._")
    L()

    # --- question_run_metrics ---
    L("### question_run_metrics")
    if table_exists(con, "question_run_metrics"):
        n = row_count(con, "question_run_metrics")
        L(f"Total rows: **{fmt_num(n)}**")
        rows = safe_query(con, """
            SELECT
                COUNT(*) AS n,
                ROUND(AVG(wall_ms), 0) AS avg_wall_ms,
                ROUND(SUM(cost_usd), 4) AS total_cost,
                ROUND(AVG(cost_usd), 4) AS avg_cost
            FROM question_run_metrics
        """)
        if rows and rows[0]:
            n, avg_wall, total_cost, avg_cost = rows[0]
            L(f"- Avg wall time: {fmt_float(avg_wall, 0)}ms")
            L(f"- Total cost: ${fmt_float(total_cost, 4)}")
            L(f"- Avg cost per question: ${fmt_float(avg_cost, 4)}")
        L()
    else:
        L("_Table does not exist._")
    L()

    # =====================================================================
    # 7. STRUCTURED DATA CONNECTORS
    # =====================================================================
    L("## 7. Structured Data Connectors")
    L()

    def _acaps_snapshot_stats(table):
        # Callable connector-table stats for the ACAPS tables whose
        # snapshot_date is a "%b%Y" VARCHAR label: SQL MIN/MAX sorted them
        # alphabetically ("Apr2026 → Sep2025"); parse chronologically instead.
        def _stats(con):
            n = row_count(con, table)
            ctys = safe_query(con, f"SELECT COUNT(DISTINCT iso3) FROM {table}")
            n_ctys = ctys[0][0] if ctys else None
            mn, mx = chrono_min_max(con, table, "snapshot_date")
            return [(n, n_ctys, mn, mx)]
        return _stats

    connector_tables = [
        ("seasonal_forecasts", """
            SELECT variable, COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   MIN(forecast_issue_date) AS min_date, MAX(forecast_issue_date) AS max_date
            FROM seasonal_forecasts GROUP BY variable ORDER BY variable
        """, ["variable", "rows", "countries", "min_date", "max_date"]),
        ("conflict_forecasts", """
            SELECT source, COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   MIN(forecast_issue_date) AS min_date, MAX(forecast_issue_date) AS max_date
            FROM conflict_forecasts GROUP BY source ORDER BY source
        """, ["source", "rows", "countries", "min_date", "max_date"]),
        ("reliefweb_reports", """
            SELECT COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries
            FROM reliefweb_reports
        """, ["rows", "countries"]),
        ("acled_political_events", """
            SELECT COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   MIN(event_date) AS min_date, MAX(event_date) AS max_date
            FROM acled_political_events
        """, ["rows", "countries", "min_date", "max_date"]),
        ("acaps_inform_severity", _acaps_snapshot_stats("acaps_inform_severity"),
         ["rows", "countries", "min_date", "max_date"]),
        ("acaps_inform_severity_trend", _acaps_snapshot_stats("acaps_inform_severity_trend"),
         ["rows", "countries", "min_date", "max_date"]),
        ("acaps_risk_radar", """
            SELECT COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   COUNT(DISTINCT risk_type) AS risk_types
            FROM acaps_risk_radar
        """, ["rows", "countries", "risk_types"]),
        ("acaps_daily_monitoring", """
            SELECT COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries,
                   MIN(entry_date) AS min_date, MAX(entry_date) AS max_date
            FROM acaps_daily_monitoring
        """, ["rows", "countries", "min_date", "max_date"]),
        ("acaps_humanitarian_access", _acaps_snapshot_stats("acaps_humanitarian_access"),
         ["rows", "countries", "min_date", "max_date"]),
    ]

    empty_connector_tables = []

    for tbl_name, query, col_names in connector_tables:
        L(f"### {tbl_name}")
        if table_exists(con, tbl_name):
            n = row_count(con, tbl_name)
            L(f"Total rows: **{fmt_num(n)}**")
            if n and n > 0:
                rows = query(con) if callable(query) else safe_query(con, query)
                if rows:
                    L()
                    L("| " + " | ".join(col_names) + " |")
                    L("|" + "---|" * len(col_names))
                    for row in rows:
                        L("| " + " | ".join(str(x) for x in row) + " |")
            else:
                empty_connector_tables.append(tbl_name)
            L()
        else:
            L("_Table does not exist._")
        L()

    L("_`ipc_phases` is a legacy table from the retired `pythia/ipc_phases.py` "
      "connector (dead code) — deliberately not tracked here; the active IPC "
      "path writes `facts_resolved` via `resolver/connectors/ipc_api.py`._")
    L()

    # --- CrisisWatch (ICG) edition detail ---
    # Explicit answer to "what month is CrisisWatch on?" — the store step
    # accumulates one edition per (year, month), so list every edition,
    # name the latest, and break down its arrows/alerts.
    L("### crisiswatch_entries (ICG CrisisWatch)")
    if table_exists(con, "crisiswatch_entries"):
        n = row_count(con, "crisiswatch_entries")
        L(f"Total rows: **{fmt_num(n)}**")
        L()
        if n:
            eds = safe_query(con, """
                SELECT year, month, COUNT(*) AS entries,
                       COUNT(DISTINCT iso3) AS countries,
                       MAX(fetched_at) AS fetched_at
                FROM crisiswatch_entries
                GROUP BY year, month
                ORDER BY year DESC, month DESC
            """)
            if eds:
                ly, lm = int(eds[0][0]), int(eds[0][1])
                L(f"**Latest edition: {ly:04d}-{lm:02d}** "
                  f"({fmt_num(eds[0][2])} entries, {eds[0][3]} countries, "
                  f"fetched {eds[0][4]}).")
                L()
                L("**All editions present (newest first):**")
                L()
                L("| edition | entries | countries | fetched_at |")
                L("|---------|--------:|----------:|------------|")
                for y, m, e, c, fa in eds:
                    L(f"| {int(y):04d}-{int(m):02d} | {fmt_num(e)} | {c} | {fa} |")
                L()
                dist = safe_query(con, f"""
                    SELECT COALESCE(NULLIF(arrow, ''), '(none)') AS arrow,
                           COALESCE(NULLIF(alert_type, ''), '(none)') AS alert_type,
                           COUNT(*) AS n
                    FROM crisiswatch_entries
                    WHERE year = {ly} AND month = {lm}
                    GROUP BY 1, 2 ORDER BY n DESC
                """)
                if dist:
                    L(f"Arrow / alert breakdown for the latest edition "
                      f"({ly:04d}-{lm:02d}):")
                    L()
                    L("| arrow | alert_type | count |")
                    L("|-------|-----------|------:|")
                    for a, al, cn in dist:
                        L(f"| {a} | {al} | {cn} |")
                    L()
    else:
        L("_Table does not exist._")
    L()

    # --- Check 1: Conflict forecast value range ---
    if table_exists(con, "conflict_forecasts") and row_count(con, "conflict_forecasts"):
        L("### Conflict Forecast Value Range Check")
        cf_stats = safe_query(con, """
            SELECT source, metric,
                   COUNT(*) AS n,
                   ROUND(MIN(value), 3) AS min_val,
                   ROUND(MAX(value), 3) AS max_val,
                   ROUND(AVG(value), 3) AS avg_val,
                   ROUND(STDDEV(value), 3) AS stddev_val
            FROM conflict_forecasts
            GROUP BY source, metric
            ORDER BY source, metric
        """)
        if cf_stats:
            cf_cols = ["source", "metric", "n", "min", "max", "avg", "stddev"]
            L("| " + " | ".join(cf_cols) + " |")
            L("|" + "---|" * len(cf_cols))
            for row in cf_stats:
                L("| " + " | ".join(str(x) for x in row) + " |")
            L()
            prob_metrics = {
                "cf_armed_conflict_risk_3m",
                "cf_armed_conflict_risk_12m",
                "cf_violence_intensity_3m",
            }
            for row in cf_stats:
                src, metric, n_rows, min_v, max_v, avg_v, std_v = row
                if metric in prob_metrics and max_v is not None and float(max_v) > 10:
                    L(f"> **WARNING:** {src} `{metric}` values appear non-probability "
                      f"(max={max_v}). Possible wrong column selection in connector.")
                    L()
        L()

    # --- Check 7: Conflict forecast per-country sample ---
    if table_exists(con, "conflict_forecasts") and row_count(con, "conflict_forecasts"):
        L("### Conflict Forecast Country Sample")
        cf_sample = safe_query(con, """
            SELECT source, iso3, metric, ROUND(value, 4) AS value
            FROM conflict_forecasts
            WHERE iso3 IN ('IRN','SOM','ETH','SDN','UKR')
            ORDER BY source, iso3, metric
        """)
        if cf_sample:
            sample_cols = ["source", "iso3", "metric", "value"]
            L("| " + " | ".join(sample_cols) + " |")
            L("|" + "---|" * len(sample_cols))
            for row in cf_sample:
                L("| " + " | ".join(str(x) for x in row) + " |")
        else:
            L("_No data for sample countries (IRN, SOM, ETH, SDN, UKR)._")
        L()

    # --- Check 5: Conflict forecast staleness ---
    if table_exists(con, "conflict_forecasts") and row_count(con, "conflict_forecasts"):
        L("### Conflict Forecast Staleness Check")
        cf_stale = safe_query(con, """
            SELECT source, MAX(forecast_issue_date) AS latest,
                   CURRENT_DATE - MAX(forecast_issue_date) AS days_old
            FROM conflict_forecasts GROUP BY source
        """)
        if cf_stale:
            stale_cols = ["source", "latest_issue_date", "days_old"]
            L("| " + " | ".join(stale_cols) + " |")
            L("|" + "---|" * len(stale_cols))
            for row in cf_stale:
                L("| " + " | ".join(str(x) for x in row) + " |")
            L()
            for row in cf_stale:
                src, latest, days_old = row
                if days_old is not None and int(days_old) > 45:
                    L(f"> **WARNING:** `{src}` last forecast is {days_old} days old (>{45} day threshold).")
                    L()
        L()

    # --- Check 2: Seasonal forecasts country list ---
    if table_exists(con, "seasonal_forecasts") and row_count(con, "seasonal_forecasts"):
        sf_countries = safe_query(con, "SELECT DISTINCT iso3 FROM seasonal_forecasts ORDER BY iso3")
        n_sf_countries = len(sf_countries)
        if n_sf_countries < 50:
            L("### Seasonal Forecasts Country Coverage")
            L(f"> **WARNING:** Only **{n_sf_countries}** countries have seasonal forecast data (expected 170+).")
            L()
            iso_list = ", ".join(r[0] for r in sf_countries)
            L(f"Countries with data: `{iso_list}`")
            L()

    # --- Check 3: Empty connector table warnings ---
    if empty_connector_tables:
        L("### Empty Connector Table Warnings")
        parts = ", ".join(f"{t} (0 rows)" for t in empty_connector_tables)
        L(f"> **WARNING: EMPTY CONNECTOR TABLES:** {parts}")
        L()

    # --- Check 4: IDMC/IDU hazard code check ---
    if table_exists(con, "facts_deltas"):
        L("### IDMC Displacement Hazard Code Check")
        idu_rows = safe_query(con, """
            SELECT hazard_code, metric, COUNT(*) AS n, COUNT(DISTINCT iso3) AS countries
            FROM facts_deltas
            WHERE lower(metric) = 'new_displacements'
            GROUP BY hazard_code, metric
            ORDER BY hazard_code
        """)
        if idu_rows:
            idu_cols = ["hazard_code", "metric", "rows", "countries"]
            L("| " + " | ".join(idu_cols) + " |")
            L("|" + "---|" * len(idu_cols))
            for row in idu_rows:
                L("| " + " | ".join(str(x) for x in row) + " |")
            L()
            hz_codes = {str(r[0]).upper().strip() for r in idu_rows}
            # IDMC writes hazard_code=IDU by design; the forecaster's
            # displacement history loaders match IN ('ACE','IDU') (see
            # forecaster/history_loaders.py), so IDU-only data is expected
            # and handled. Only warn on genuinely unexpected codes.
            unexpected = hz_codes - {"IDU", "ACE"}
            if "IDU" in hz_codes:
                L("_Note: IDMC rows use hazard_code=IDU by design; the "
                  "forecaster matches `IN ('ACE','IDU')`, so this needs no "
                  "action._")
                L()
            if unexpected:
                L(f"> **WARNING:** unexpected hazard codes on new_displacements "
                  f"rows: {', '.join(sorted(unexpected))} — the forecaster only "
                  f"matches ACE/IDU.")
                L()
        else:
            L("_No new_displacements rows in facts_deltas._")
        L()

    # --- Check 6: HDX Signals note ---
    L("### HDX Signals")
    L("_Note: HDX Signals data is cached as a local CSV file, not stored in DuckDB. "
      "HDX Signals health cannot be checked from the DB alone. "
      "Check the `hs_country_evidence` artifact for HDX Signals availability._")
    L()

    # =====================================================================
    # 8. GAME-THEORETIC & PREDICTION MARKET TABLES
    # =====================================================================
    L("## 8. Game-Theoretic & Prediction Market Tables")
    L()

    for tbl_name in ["gtmc1_runs", "gtmc1_actors", "pm_checks"]:
        L(f"### {tbl_name}")
        if table_exists(con, tbl_name):
            n = row_count(con, tbl_name)
            L(f"Total rows: **{fmt_num(n)}**")
            if n and n > 0:
                schema = schema_info(con, tbl_name)
                L()
                L("Schema:")
                L("```")
                for row in schema:
                    L(f"  {row[1]:30s} {row[2]}")
                L("```")
                L()
                sample = safe_query(con, f"SELECT * FROM {tbl_name} LIMIT 3")
                if sample:
                    col_names = [d[0] for d in con.description]
                    L("Sample rows:")
                    L("```")
                    for row in sample:
                        L(str(dict(zip(col_names, row))))
                    L("```")
            L()
        else:
            L("_Table does not exist._")
        L()

    # =====================================================================
    # 9. PROVENANCE & METADATA
    # =====================================================================
    L("## 9. Provenance & Metadata")
    L()

    for tbl_name in ["run_provenance", "meta_runs", "manifests", "snapshots"]:
        L(f"### {tbl_name}")
        if table_exists(con, tbl_name):
            n = row_count(con, tbl_name)
            L(f"Total rows: **{fmt_num(n)}**")
            if n and n > 0:
                sample = safe_query(con, f"SELECT * FROM {tbl_name} ORDER BY ROWID DESC LIMIT 5")
                if sample:
                    col_names = [d[0] for d in con.description]
                    L()
                    L("Recent entries:")
                    L("```")
                    for row in sample:
                        L(str(dict(zip(col_names, row))))
                    L("```")
            L()
        else:
            L("_Table does not exist._")
        L()

    # =====================================================================
    # 10. SCHEMA DETAIL FOR ALL TABLES
    # =====================================================================
    L("## 10. Full Schema Reference")
    L()
    L("<details>")
    L("<summary>Click to expand full column definitions for all tables</summary>")
    L()

    for (tbl,) in all_tables:
        L(f"### {tbl}")
        schema = schema_info(con, tbl)
        if schema:
            L()
            L("| # | column | type | notnull | default | pk |")
            L("|--:|--------|------|:-------:|---------|:--:|")
            for row in schema:
                cid, name, ctype, notnull, default, pk = row
                L(f"| {cid} | `{name}` | {ctype} | {'Y' if notnull else ''} | {default or ''} | {'Y' if pk else ''} |")
            L()
        else:
            L("_(no schema info)_")
            L()

    L("</details>")
    L()

    # =====================================================================
    # 11. DATABASE-WIDE STATISTICS
    # =====================================================================
    L("## 11. Database-Wide Statistics")
    L()

    total_rows = 0
    for (tbl,) in all_tables:
        n = row_count(con, tbl)
        if n:
            total_rows += n

    L(f"- **Total tables:** {len(all_tables)}")
    L(f"- **Total rows across all tables:** {fmt_num(total_rows)}")
    L(f"- **Database file size:** {db_size_mb:.1f} MB")
    L()

    # Country coverage
    try:
        rows = con.execute("""
            SELECT COUNT(DISTINCT iso3) FROM (
                SELECT DISTINCT iso3 FROM facts_deltas
                UNION
                SELECT DISTINCT iso3 FROM hs_triage
                UNION
                SELECT DISTINCT iso3 FROM questions
            )
        """).fetchone()
        if rows:
            L(f"- **Unique countries (across facts_deltas + hs_triage + questions):** {rows[0]}")
    except Exception:
        pass

    # Hazard coverage
    try:
        rows = con.execute("""
            SELECT DISTINCT hazard_code
            FROM hs_triage
            ORDER BY hazard_code
        """).fetchall()
        if rows:
            hazards = [r[0] for r in rows]
            L(f"- **Hazard codes in hs_triage:** {', '.join(hazards)}")
    except Exception:
        pass

    L()

    con.close()

    return "\n".join(lines)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Resolver DuckDB comprehensive inspection report",
    )
    parser.add_argument("--db", default="data/resolver.duckdb", help="Path to resolver.duckdb")
    parser.add_argument("--out", default="resolver_inspect.md", help="Output Markdown path")
    parser.add_argument(
        "--country-list",
        default=str(DEFAULT_COUNTRY_LIST),
        help="HS target country list (one name per line) for the inject-readiness section",
    )
    parser.add_argument(
        "--countries-csv",
        default=str(DEFAULT_COUNTRIES_CSV),
        help="ISO registry CSV (country_name,iso3) used to resolve the country list",
    )
    args = parser.parse_args(argv)

    report = build_report(
        Path(args.db),
        country_list_path=Path(args.country_list),
        countries_csv_path=Path(args.countries_csv),
    )
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(report.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
