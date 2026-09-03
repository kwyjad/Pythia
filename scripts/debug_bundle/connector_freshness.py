# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""How old every structured data source was when this run started.

Staleness is the failure mode this pipeline suffers most and notices
least: a connector that fetches successfully and returns last quarter's
edition looks identical, from every log and every green check, to one
that is current. In September the CrisisWatch table held February and June
and nothing else, and the forecast prompts for October to March carried the
June edition labelled "(June 2026)" with no caveat.

Two things the plain row count cannot say, and both are here.

Age is measured against the OBSERVATION, never the fetch — a row rewritten
every run keeps its fetch date current while the reading it carries ages.
And a source is asked whether its own staleness reaches the PROMPT: a
figure that is stale and labelled stale is a different situation from one
that is stale and presented as current, and only the second is a data
error.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

# (label, table, observation-timestamp column, extra WHERE, iso3 column,
#  warn_days, stale_days, prompt-staleness mechanism)
#
# The thresholds follow each source's own publication cadence: a monthly
# publication is not late at 40 days and a daily conflict feed is very late
# at 30. "prompt_mechanism" names the code that labels the age in the
# prompt, or says none exists.
SOURCES: tuple[tuple[str, str, str, str, str, int, int, str], ...] = (
    ("ACLED fatalities", "acled_monthly_fatalities", "month", "", "iso3", 45, 75,
     "none"),
    ("IDMC displacement", "facts_deltas", "as_of_date",
     "WHERE metric = 'new_displacements'", "iso3", 60, 120, "none"),
    ("IFRC PA", "facts_resolved", "as_of_date",
     "WHERE UPPER(publisher) = 'IFRC'", "iso3", 60, 120, "none"),
    ("GDACS", "facts_resolved", "as_of_date",
     "WHERE metric = 'event_occurrence'", "iso3", 30, 60, "none"),
    ("FEWS NET IPC", "facts_resolved", "as_of_date",
     "WHERE metric = 'phase3plus_in_need' AND UPPER(publisher) = 'FEWS NET'",
     "iso3", 90, 180, "none"),
    ("IPC API", "facts_resolved", "as_of_date",
     "WHERE metric = 'phase3plus_in_need' AND UPPER(publisher) = 'IPC'",
     "iso3", 120, 240, "none"),
    ("Conflict forecasts", "conflict_forecasts", "forecast_issue_date", "", "iso3",
     45, 90, "conflict_forecasts.age_note (states the actual age at >=2 months)"),
    ("ReliefWeb", "reliefweb_reports", "published_date", "", "iso3", 30, 60, "none"),
    ("ACAPS INFORM Severity", "acaps_inform_severity", "snapshot_date", "", "iso3", 60, 120, "none"),
    ("ACAPS Risk Radar", "acaps_risk_radar", "fetched_at", "", "iso3", 60, 120, "none"),
    ("ACAPS Daily Monitoring", "acaps_daily_monitoring", "entry_date", "", "iso3", 30, 60, "none"),
    ("ACAPS Humanitarian Access", "acaps_humanitarian_access", "snapshot_date", "", "iso3", 90, 180, "none"),
    ("ACLED political events", "acled_political_events", "event_date", "", "iso3", 30, 60, "none"),
    ("GDELT", "gdelt_conflict_indicators", "event_date", "", "iso3", 14, 30, "none"),
    ("HDX Signals", "hdx_signals", "signal_date", "", "iso3", 60, 120, "none"),
    ("NMME seasonal", "seasonal_forecasts", "forecast_issue_date", "", "iso3", 60, 120, "none"),
    ("ENSO", "enso_state", "observation_date", "", "", 60, 120,
     "enso_module carries the last good record forward with a STALE READING line"),
    ("Seasonal TC", "seasonal_tc_outlooks", "fetched_at", "", "", 200, 400, "none"),
    ("IPC phases (legacy, unused)", "ipc_phases", "analysis_date", "", "iso3", 0, 0, "none"),
    # crisiswatch_entries carries no date column at all — its observation is
    # the EDITION, which is (year, month), so its age is computed from that
    # rather than from a column. See crisiswatch_detail.
    ("CrisisWatch", "crisiswatch_entries", "", "", "iso3", 60, 90,
     "crisiswatch.format_crisiswatch_for_prompt (STALENESS WARNING at >=3 editions)"),
)

# Sources whose age the prompt states. Everything else presents its figure
# with no label, which is why the age column here is the only place it
# shows up at all.
_LABELS_AGE_IN_PROMPT = {"Conflict forecasts", "ENSO", "CrisisWatch"}


def _columns(con, table: str) -> set[str]:
    try:
        # table_info yields (cid, name, ...) — the NAME is column 1, and reading
        # column 0 gives the ordinal, so every lookup silently missed.
        return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _table_exists(con, table: str) -> bool:
    return bool(_columns(con, table))


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # ``ym`` columns are YYYY-MM; treat them as the first of the month, which
    # under-states the age rather than over-stating it.
    if len(text) == 7 and text[4] == "-":
        text = text + "-01"
    # Several of these columns are VARCHAR carrying an ISO timestamp with a
    # zone (reliefweb_reports.published_date, the ACAPS fetched_at fields),
    # so try the full parse first and fall back to the leading date.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _pick_observation_column(con, table: str, preferred: str) -> str | None:
    """The column that dates the OBSERVATION, not the fetch.

    A fetch column is the fallback, and when it is all there is the row
    says so — measuring staleness from ``fetched_at`` is how a source
    rewritten every run reads as fresh forever.
    """

    cols = _columns(con, table)
    if preferred and preferred in cols:
        return preferred
    for candidate in ("as_of_date", "observation_date", "event_date", "date",
                      "signal_date", "forecast_issue_date", "published_date",
                      "snapshot_date", "entry_date", "analysis_date", "month",
                      "ym", "period_start"):
        if candidate in cols:
            return candidate
    for candidate in ("created_at", "fetched_at", "retrieved_at", "updated_at", "timestamp"):
        if candidate in cols:
            return candidate
    return None


_FETCH_COLUMNS = {"created_at", "fetched_at", "retrieved_at", "updated_at", "timestamp"}


def collect(
    con,
    *,
    countries: list[str] | None = None,
    questions: list[dict[str, Any]] | None = None,
    run_start: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (rows, crisiswatch_detail). Never raises."""

    today = run_start or datetime.now(timezone.utc).date()
    target = [str(c).upper() for c in (countries or [])]
    target_set = set(target)
    rows: list[dict[str, Any]] = []

    for (label, table, ts_col, where, iso_col, warn_days, stale_days, mechanism) in SOURCES:
        row: dict[str, Any] = {
            "source": label,
            "table": table,
            "row_count": 0,
            "observation_column": "",
            "max_data_timestamp": "",
            "age_days_at_run_start": "",
            "warn_threshold_days": warn_days,
            "stale_threshold_days": stale_days,
            "status": "absent",
            "countries_covered": 0,
            "countries_missing": 0,
            "missing_iso3s": "",
            "prompt_staleness_warning": "no mechanism",
            "prompt_staleness_mechanism": mechanism,
            "note": "",
        }
        if not _table_exists(con, table):
            row["note"] = "table absent"
            rows.append(row)
            continue

        try:
            row["row_count"] = int(
                con.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
            )
        except Exception as exc:  # noqa: BLE001
            row["note"] = f"count failed: {exc}"

        column = _pick_observation_column(con, table, ts_col)
        if column:
            row["observation_column"] = column
            if column in _FETCH_COLUMNS:
                row["note"] = (
                    (row["note"] + "; ") if row["note"] else ""
                ) + f"age measured from {column} (a fetch column — no observation date on this table)"
            try:
                value = con.execute(f"SELECT MAX({column}) FROM {table} {where}").fetchone()[0]
                observed = _to_date(value)
                if observed is not None:
                    row["max_data_timestamp"] = observed.isoformat()
                    row["age_days_at_run_start"] = (today - observed).days
            except Exception as exc:  # noqa: BLE001
                row["note"] = ((row["note"] + "; ") if row["note"] else "") + f"max({column}) failed: {exc}"

        if target and iso_col and iso_col in _columns(con, table):
            try:
                ph = ",".join("?" for _ in target)
                clause = f"{where} AND" if where else "WHERE"
                covered = {
                    str(r[0]).upper()
                    for r in con.execute(
                        f"SELECT DISTINCT UPPER({iso_col}) FROM {table} "
                        f"{clause} UPPER({iso_col}) IN ({ph})",
                        target,
                    ).fetchall()
                }
                missing = sorted(target_set - covered)
                row["countries_covered"] = len(covered)
                row["countries_missing"] = len(missing)
                row["missing_iso3s"] = " ".join(missing)
            except Exception as exc:  # noqa: BLE001
                row["note"] = ((row["note"] + "; ") if row["note"] else "") + f"coverage failed: {exc}"

        if label == "CrisisWatch":
            # The edition IS the observation: a table rewritten every run
            # keeps fetched_at current while the edition it holds ages, and
            # in September that edition was three months old.
            _cw_age(con, row, today)

        age = row["age_days_at_run_start"]
        if row["row_count"] == 0:
            row["status"] = "empty"
        elif age == "":
            row["status"] = "unknown"
        elif stale_days and age >= stale_days:
            row["status"] = "stale"
        elif warn_days and age >= warn_days:
            row["status"] = "warn"
        else:
            row["status"] = "fresh"

        if label in _LABELS_AGE_IN_PROMPT:
            row["prompt_staleness_warning"] = (
                "yes" if row["status"] in ("warn", "stale") else "not needed"
            )
        rows.append(row)

    return rows, crisiswatch_detail(con, questions=questions, today=today)


def _cw_age(con, row: dict[str, Any], today: date) -> None:
    """Age the CrisisWatch row by its newest EDITION, not by fetched_at."""

    try:
        latest = con.execute(
            "SELECT MAX(year * 100 + COALESCE(month, 0)) FROM crisiswatch_entries"
        ).fetchone()[0]
    except Exception:
        return
    if not latest:
        return
    year, month = int(latest) // 100, int(latest) % 100
    if not (1 <= month <= 12):
        return
    # A CrisisWatch edition covers month M and is published early in M+1, so
    # date it from the END of the month it covers.
    end = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    row["observation_column"] = "edition (year, month)"
    row["max_data_timestamp"] = f"{year}-{month:02d}"
    row["age_days_at_run_start"] = (today - end).days
    row["note"] = ""


def crisiswatch_detail(
    con, *, questions: list[dict[str, Any]] | None = None, today: date | None = None
) -> dict[str, Any]:
    """CrisisWatch's editions and its ACE inject status.

    "Latest edition 2026-06" hid a five-month gap: the table held 2026-02
    and 2026-06, so the per-country rows were a mix of two editions four
    months apart and no single field said so. Every edition present is
    listed, with how many countries each one supplies.

    The ACE half asks the question the forecast actually depends on: how
    many ACE questions were forecast for a country with no CrisisWatch row
    at all, and therefore no conflict arrow in their prompts.
    """

    today = today or datetime.now(timezone.utc).date()
    out: dict[str, Any] = {"available": False}
    if not _table_exists(con, "crisiswatch_entries"):
        out["note"] = "crisiswatch_entries table absent"
        return out
    try:
        rows = [
            dict(zip(("iso3", "arrow", "alert_type", "year", "month"), r))
            for r in con.execute(
                "SELECT UPPER(iso3), COALESCE(arrow,''), COALESCE(alert_type,''), "
                "year, month FROM crisiswatch_entries"
            ).fetchall()
        ]
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"read failed: {exc}"
        return out

    out["available"] = True
    out["n_rows"] = len(rows)
    editions: dict[str, int] = {}
    for r in rows:
        year, month = r.get("year"), r.get("month")
        if not year:
            continue
        label = f"{int(year)}-{int(month):02d}" if month else str(int(year))
        editions[label] = editions.get(label, 0) + 1
    out["editions_present"] = [
        {"edition": k, "n_countries": v} for k, v in sorted(editions.items())
    ]
    out["n_editions"] = len(editions)
    if editions:
        latest = sorted(editions)[-1]
        out["latest_edition"] = latest
        try:
            y, m = latest.split("-")
            out["latest_edition_age_months"] = (today.year - int(y)) * 12 + (today.month - int(m))
        except Exception:
            out["latest_edition_age_months"] = None
        # Two editions three months apart in one table is not a fresher
        # edition arriving; it is months that never landed.
        if len(editions) > 1:
            first = sorted(editions)[0]
            out["edition_span"] = f"{first} .. {latest}"

    arrows: dict[str, int] = {}
    for r in rows:
        arrows[r["arrow"] or "(none)"] = arrows.get(r["arrow"] or "(none)", 0) + 1
    out["arrow_counts"] = arrows
    out["n_countries_with_arrow"] = sum(1 for r in rows if r["arrow"])
    out["n_countries_without_arrow"] = sum(1 for r in rows if not r["arrow"])
    out["n_alerts"] = sum(1 for r in rows if r["alert_type"])
    alerts: dict[str, int] = {}
    for r in rows:
        if r["alert_type"]:
            alerts[r["alert_type"]] = alerts.get(r["alert_type"], 0) + 1
    out["alert_counts"] = alerts

    covered = {r["iso3"] for r in rows if r["iso3"]}
    ace_countries = sorted(
        {
            str(q.get("iso3") or "").upper()
            for q in (questions or [])
            if str(q.get("hazard_code") or "").upper() == "ACE"
        }
    )
    missing = [c for c in ace_countries if c and c not in covered]
    out["ace_countries_forecast"] = len(ace_countries)
    out["ace_countries_without_crisiswatch_row"] = len(missing)
    out["ace_countries_missing_iso3s"] = missing
    out["n_ace_questions_without_crisiswatch"] = sum(
        1
        for q in (questions or [])
        if str(q.get("hazard_code") or "").upper() == "ACE"
        and str(q.get("iso3") or "").upper() in set(missing)
    )
    return out
