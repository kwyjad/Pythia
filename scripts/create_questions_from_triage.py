# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from pythia.db.schema import ensure_schema


DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"
LOG = logging.getLogger(__name__)


@dataclass
class TriagedHazard:
    iso3: str
    hazard_code: str
    tier: str
    triage_score: float
    need_full_spd: bool
    track: Optional[int] = None


SUPPORTED_HAZARD_METRICS: Dict[str, List[str]] = {
    "ACE": ["FATALITIES", "PA"],
    "CU": ["PA"],
    "DR": ["PA"],
    "FL": ["PA"],
    "TC": ["PA"],
    "HW": ["PA"],
    "DI": ["PA"],
}

HAZARD_HUMAN_NAMES = {
    "DR": "drought",
    "FL": "flooding",
    "HW": "heatwave",
    "TC": "tropical cyclone",
}

COUNTRY_NAMES = {
    "ETH": "Ethiopia",
    "SOM": "Somalia",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Pythia questions from hs_triage for the latest HS run."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_URL,
        help="DuckDB URL, e.g. duckdb:///data/resolver.duckdb",
    )
    parser.add_argument(
        "--hs-run-id",
        default=None,
        help="Optional explicit hs_run_id. If omitted, we use the latest from hs_runs/hs_triage.",
    )
    return parser.parse_args()


def _resolve_db_path(db_url: str) -> str:
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///") :]
    return db_url


def _select_hs_run_id(
    con: duckdb.DuckDBPyConnection, explicit: Optional[str]
) -> Optional[str]:
    if explicit:
        return explicit

    row = con.execute(
        """
        SELECT hs_run_id
        FROM hs_runs
        ORDER BY generated_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return row[0]

    row = con.execute(
        """
        SELECT run_id
        FROM hs_triage
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return row[0]

    return None


def _metrics_for_hazard(hz: str) -> List[str]:
    hz_up = (hz or "").upper()
    return SUPPORTED_HAZARD_METRICS.get(hz_up, [])


def _country_label(iso3: str) -> str:
    return COUNTRY_NAMES.get((iso3 or "").upper(), (iso3 or "").upper())


def _build_question_wording(
    iso3: str,
    hazard_code: str,
    metric: str,
    window_start_date: date,
    window_end_date: date,
) -> str:
    iso3_up = (iso3 or "").upper()
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    country = _country_label(iso3_up)

    start_str = window_start_date.isoformat()
    end_str = window_end_date.isoformat()

    if hz == "ACE" and m == "PA":
        return (
            f"How many people will be displaced each month by armed conflict in {country} "
            f"between {start_str} and {end_str}, as resolved by IDMC?"
        )

    if hz == "ACE" and m == "FATALITIES":
        return (
            f"How many people will be killed each month by armed conflict in {country} "
            f"between {start_str} and {end_str}, as resolved by ACLED?"
        )

    if hz == "DI" and m == "PA":
        return (
            f"How many people will enter {country} because of armed conflict in a neighbouring country "
            f"between {start_str} and {end_str}?"
        )

    if hz in {"DR", "FL", "HW", "TC"} and m == "PA":
        hazard_name = HAZARD_HUMAN_NAMES.get(hz, hz)
        return (
            f"How many people will be affected each month by {hazard_name} in {country} "
            f"between {start_str} and {end_str}, as resolved by EM-DAT?"
        )

    return (
        f"How many people will be affected each month in {country} by hazard {hz} "
        f"between {start_str} and {end_str}?"
    )


def _compute_target_and_window(today: date) -> Tuple[str, date, date]:
    start_month = today.month + 1
    start_year = today.year + (start_month - 1) // 12
    start_month = ((start_month - 1) % 12) + 1
    opening = date(start_year, start_month, 1)

    end_month = start_month + 5
    end_year = start_year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    if end_month == 12:
        next_year, next_month = end_year + 1, 1
    else:
        next_year, next_month = end_year, end_month + 1
    closing = date(next_year, next_month, 1) - timedelta(days=1)

    target_month_label = f"{opening.year:04d}-{opening.month:02d}"
    return target_month_label, opening, closing


def _load_triage_rows(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> List[TriagedHazard]:
    # Check if track column exists (backward compat with older DBs)
    cols = {r[0] for r in con.execute("DESCRIBE hs_triage").fetchall()}
    track_expr = "track" if "track" in cols else "NULL AS track"
    rows = con.execute(
        f"""
        SELECT iso3, hazard_code, tier, triage_score, need_full_spd, {track_expr}
        FROM hs_triage
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()

    triaged: List[TriagedHazard] = []
    for iso3, hz, tier, score, need_full_spd, track_val in rows:
        iso3_up = (iso3 or "").upper()
        hz_up = (hz or "").upper()
        tier_str = (tier or "").lower()
        need = bool(need_full_spd) or tier_str == "priority"
        score_f = float(score or 0.0)
        track_int = int(track_val) if track_val is not None else None
        if hz_up == "ACO":
            LOG.info(
                "Skipping ACO triage/question for %s; ACE is the canonical conflict hazard",
                iso3_up,
            )
            continue
        if not need:
            continue
        if not _metrics_for_hazard(hz_up):
            continue
        triaged.append(
            TriagedHazard(
                iso3=iso3_up,
                hazard_code=hz_up,
                tier=tier_str,
                triage_score=score_f,
                need_full_spd=need,
                track=track_int,
            )
        )

    triaged.sort(key=lambda th: (th.iso3, th.hazard_code, th.tier, th.triage_score))
    return triaged


def _upsert_question(
    con: duckdb.DuckDBPyConnection,
    *,
    question_id: str,
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    wording: str,
    status: str,
    metadata: Dict[str, Any],
    target_month: Optional[str],
    window_start_date: date,
    window_end_date: date,
    track: Optional[int] = None,
) -> None:
    meta_json = json.dumps(metadata, ensure_ascii=False)
    con.execute("DELETE FROM questions WHERE question_id = ?", [question_id])
    con.execute(
        """
        INSERT INTO questions (
            question_id, hs_run_id, scenario_ids_json,
            iso3, hazard_code, metric,
            target_month, window_start_date, window_end_date,
            wording, status, pythia_metadata_json, track
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            question_id,
            hs_run_id,
            "[]",
            iso3,
            hazard_code,
            metric,
            target_month,
            window_start_date,
            window_end_date,
            wording,
            status,
            meta_json,
            track,
        ],
    )


def create_questions_from_triage(db_url: str, hs_run_id: Optional[str] = None) -> int:
    db_path = _resolve_db_path(db_url)
    con = duckdb.connect(db_path)
    try:
        ensure_schema(con)
        run_id = _select_hs_run_id(con, hs_run_id)
        if not run_id:
            print("create_questions_from_triage: no hs_run_id found; nothing to do.")
            return 0

        triaged = _load_triage_rows(con, run_id)
        if not triaged:
            print(f"create_questions_from_triage: no eligible hs_triage rows for run_id={run_id}.")
            return 0

        inserted = 0
        today = date.today()
        target_month, opening, closing = _compute_target_and_window(today)

        for th in triaged:
            metrics = _metrics_for_hazard(th.hazard_code)
            if th.hazard_code == "ACE":
                conflict_meta = {
                    "source": "hs_triage",
                    "hs_run_id": run_id,
                    "tier": th.tier,
                    "triage_score": th.triage_score,
                    "hazard_family": "conflict",
                }
                _upsert_question(
                    con,
                    question_id=f"{th.iso3}_ACE_FATALITIES",
                    hs_run_id=run_id,
                    iso3=th.iso3,
                    hazard_code="ACE",
                    metric="FATALITIES",
                    wording=(
                        f"How many people will be killed each month by armed conflict in {th.iso3} "
                        "between the forecast start and end dates, as resolved by ACLED?"
                    ),
                    status="active",
                    metadata=conflict_meta,
                    target_month=target_month,
                    window_start_date=opening,
                    window_end_date=closing,
                    track=th.track,
                )
                inserted += 1

                _upsert_question(
                    con,
                    question_id=f"{th.iso3}_ACE_PA",
                    hs_run_id=run_id,
                    iso3=th.iso3,
                    hazard_code="ACE",
                    metric="PA",
                    wording=(
                        f"How many people will be newly displaced or affected by conflict in {th.iso3} "
                        "each month, as measured by IDMC displacement data and related sources?"
                    ),
                    status="active",
                    metadata=conflict_meta,
                    target_month=target_month,
                    window_start_date=opening,
                    window_end_date=closing,
                    track=th.track,
                )
                inserted += 1
                continue

            for mt in metrics:
                wording = _build_question_wording(
                    th.iso3, th.hazard_code, mt, opening, closing
                )
                meta = {
                    "source": "hs_triage",
                    "hs_run_id": run_id,
                    "tier": th.tier,
                    "triage_score": th.triage_score,
                }

                _upsert_question(
                    con,
                    question_id=f"{th.iso3}_{th.hazard_code}_{mt}",
                    hs_run_id=run_id,
                    iso3=th.iso3,
                    hazard_code=th.hazard_code,
                    metric=mt,
                    wording=wording,
                    status="active",
                    metadata=meta,
                    target_month=target_month,
                    window_start_date=opening,
                    window_end_date=closing,
                    track=th.track,
                )
                inserted += 1

        print(f"create_questions_from_triage: ensured {inserted} questions for run_id={run_id}")
        return inserted
    finally:
        con.close()


def main() -> None:
    args = _parse_args()
    create_questions_from_triage(args.db, args.hs_run_id)


if __name__ == "__main__":
    main()
