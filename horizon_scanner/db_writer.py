from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable

from pythia.db.schema import connect, ensure_schema
from pythia.utils.ids import scenario_id as sid, question_id as qid

HAZARD_CONFIG = {
    "FL": {
        "label": "Flood",
        "class": "Natural",
        "metric_default": "PA",
        "question_type": "emdat_pa",
        "question_template": (
            "How many people will be affected by riverine or flash flooding "
            "in {country_name} between {start_date} and {end_date}, as reported by EM-DAT?"
        ),
        "resolution_source": "EM-DAT",
    },
    "DR": {
        "label": "Drought",
        "class": "Natural",
        "metric_default": "PA",
        "question_type": "emdat_pa",
        "question_template": (
            "How many people will be affected by drought in {country_name} "
            "between {start_date} and {end_date}, as reported by EM-DAT?"
        ),
        "resolution_source": "EM-DAT",
    },
    "TC": {
        "label": "Tropical Cyclone",
        "class": "Natural",
        "metric_default": "PA",
        "question_type": "emdat_pa",
        "question_template": (
            "How many people will be affected by tropical cyclones in {country_name} "
            "between {start_date} and {end_date}, as reported by EM-DAT?"
        ),
        "resolution_source": "EM-DAT",
    },
    "HW": {
        "label": "Heat Wave",
        "class": "Natural",
        "metric_default": "PA",
        "question_type": "emdat_pa",
        "question_template": (
            "How many people will be affected by heat waves in {country_name} "
            "between {start_date} and {end_date}, as reported by EM-DAT?"
        ),
        "resolution_source": "EM-DAT",
    },
    "ACO": {
        "label": "Armed Conflict Onset",
        "class": "Conflict",
        "question_types": ["acled_fatalities", "conflict_displacement"],
    },
    "ACE": {
        "label": "Armed Conflict Escalation",
        "class": "Conflict",
        "question_types": ["acled_fatalities", "conflict_displacement"],
    },
    "CU": {
        "label": "Civil Unrest",
        "class": "Conflict/Social",
        "question_types": ["unrest_displacement"],
        "blocked": True,
    },
    "DI": {
        "label": "Displacement Influx",
        "class": "Displacement",
        "question_types": ["influx_displacement"],
    },
    "EC": {
        "label": "Economic Crisis",
        "class": "Economic",
        "blocked": True,
    },
    "PHE": {
        "label": "Public Health Emergency",
        "class": "Health",
        "blocked": True,
    },
    "MULTI": {
        "label": "Multi-driver Food Insecurity / Complex",
        "class": "Multi/Complex",
        "blocked": True,
    },
}

for _alias in ["CONFLICT", "POLITICAL_VIOLENCE", "CIVIL_CONFLICT", "URBAN_CONFLICT"]:
    HAZARD_CONFIG.setdefault(
        _alias,
        {
            "label": "Armed Conflict",
            "class": "Conflict",
            "question_types": ["acled_fatalities", "conflict_displacement"],
        },
    )

DISPLACEMENT_TEMPLATES = {
    "acled_fatalities": (
        "How many battle-related fatalities will ACLED record in {country_name} "
        "between {start_date} and {end_date}?"
    ),
    "conflict_displacement": (
        "How many people will be internally displaced by armed conflict in {country_name} "
        "between {start_date} and {end_date}, as recorded by the Internal Displacement "
        "Monitoring Center (IDMC) with the IOM Displacement Tracking Matrix (DTM) as back-up?"
    ),
    "unrest_displacement": (
        "How many people will be internally displaced by civil unrest in {country_name} "
        "between {start_date} and {end_date}, as recorded by the Internal Displacement "
        "Monitoring Center (IDMC) with the IOM Displacement Tracking Matrix (DTM) as back-up?"
    ),
    "influx_displacement": (
        "How many people from a neighbouring country will enter {country_name} between {start_date} "
        "and {end_date} because of armed conflict or other political violence, as recorded by the "
        "Internal Displacement Monitoring Center (IDMC) with the IOM Displacement Tracking Matrix (DTM) as back-up?"
    ),
}

BLOCKED_HAZARDS = {code for code, cfg in HAZARD_CONFIG.items() if cfg.get("blocked")}
BLOCKED_HAZARDS.add("ACO")
ALLOWED = set(HAZARD_CONFIG.keys())
CONFLICT_HAZARDS = {"CONFLICT", "POLITICAL_VIOLENCE", "CIVIL_CONFLICT", "URBAN_CONFLICT"}


def _to_target_month(today: date, months_ahead: int) -> str:
    y, m = today.year, today.month + months_ahead
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return f"{y:04d}-{m:02d}"


def _compute_forecast_window(today: date, target_month: str) -> tuple[date, date]:
    """
    Compute the [opening_date, closing_date] for the Pythia HS forecast questions.

    - opening_date: first day of the calendar month immediately after `today`
      (e.g. if today is 2025-11-27 → 2025-12-01;
             if today is any date in December 2025 → 2026-01-01).
    - closing_date: last day of the `target_month` (YYYY-MM) as computed by
      `_to_target_month(today, horizon_months)`.

    If `target_month` cannot be parsed, we fall back to a 6-month inclusive window
    starting at `opening_date` (i.e. last day of the month 5 months after opening).
    """
    # Opening: first day of the month after `today`
    year, month = today.year, today.month
    if month == 12:
        opening_year, opening_month = year + 1, 1
    else:
        opening_year, opening_month = year, month + 1
    opening = date(opening_year, opening_month, 1)

    # Closing: last day of target_month if parseable; otherwise 6-month fallback
    try:
        parts = target_month.split("-")
        tgt_year = int(parts[0])
        tgt_month = int(parts[1])
        if tgt_month == 12:
            closing = date(tgt_year, 12, 31)
        else:
            closing = date(tgt_year, tgt_month + 1, 1) - timedelta(days=1)
    except Exception:
        # Fallback: last day of month 5 months after opening (6 months inclusive)
        m = opening_month + 5
        y = opening_year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        if m == 12:
            closing = date(y, 12, 31)
        else:
            closing = date(y, m + 1, 1) - timedelta(days=1)

    return opening, closing


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_question_row(
    *,
    hs_run_id: str,
    scenario: dict,
    scenario_ids: list[str],
    iso3: str,
    hazard_code: str,
    hazard_cfg: dict,
    metric: str,
    wording: str,
    target_month: str,
    opening_date: date,
    closing_date: date,
    question_type: str | None,
    resolution_source: str | None,
    is_test_mode: bool,
) -> dict:
    hazard_label = hazard_cfg.get("label") or scenario.get("hazard_label") or hazard_code
    hazard_class = hazard_cfg.get("class") or ""

    best_guess = scenario.get("best_guess") or {}
    best_guess_value = float(best_guess.get(metric) or 0)

    question_id = qid(iso3, hazard_code, metric, target_month, wording)

    meta = {
        "source": "HS",
        "purpose": "hs_pipeline",
        "hazard_code": hazard_code,
        "hazard_label": hazard_label,
        "hazard_class": hazard_class,
        "question_type": question_type,
        "resolution_source": resolution_source,
        "hs_likely_month": scenario.get("likely_window_month", ""),
        "hs_probability_pct": float(scenario.get("probability_pct") or 0.0),
        "hs_run_id": hs_run_id,
        "scenario_ids": scenario_ids,
        "window_start": opening_date.isoformat(),
        "window_end": closing_date.isoformat(),
    }

    if is_test_mode:
        meta["test_mode"] = True
        meta["run_profile"] = "test"

    hs_json_raw = {"source": "HS", "raw": scenario.get("json", {})}
    if is_test_mode and isinstance(hs_json_raw.get("raw"), dict):
        hs_json_raw["raw"] = {**hs_json_raw["raw"], "test_mode": True}

    return {
        "question_id": question_id,
        "scenario_ids": scenario_ids,
        "hs_run_id": hs_run_id,
        "iso3": iso3,
        "hazard_code": hazard_code,
        "hazard_label": hazard_label,
        "metric": metric,
        "target_month": target_month,
        "window_start_date": opening_date,
        "window_end_date": closing_date,
        "wording": wording,
        "best_guess_value": best_guess_value,
        "hs_json": hs_json_raw,
        "pythia_metadata": meta,
        "status": "active",
    }


def _build_questions_for_scenario(
    *,
    hs_run_id: str,
    scenario: dict,
    target_month: str,
    opening_date: date,
    closing_date: date,
    opening_str: str,
    closing_str: str,
    is_test_mode: bool,
) -> list[dict]:
    hazard_code = (scenario.get("hazard_code") or "").upper()
    cfg = HAZARD_CONFIG.get(hazard_code, {})

    if cfg.get("blocked") or hazard_code in BLOCKED_HAZARDS:
        logging.info("[hs] Blocking question for hazard=%s (no resolution yet).", hazard_code)
        return []

    country_name = scenario.get("country_name") or scenario.get("iso3") or ""
    iso3 = (scenario.get("iso3") or "").upper()
    scenario_id = scenario.get("scenario_id")
    questions: list[dict] = []

    if not cfg:
        hazard_label = scenario.get("hazard_label") or hazard_code
        hazard_label_lower = hazard_label.lower() if isinstance(hazard_label, str) else str(hazard_label).lower()
        fallback_wording = (
            f"How many people in {country_name} will be affected by {hazard_label_lower} "
            f"between {opening_str} and {closing_str}?"
        )
        questions.append(
            _build_question_row(
                hs_run_id=hs_run_id,
                scenario=scenario,
                scenario_ids=[scenario_id],
                iso3=iso3,
                hazard_code=hazard_code,
                hazard_cfg={"label": hazard_label, "class": ""},
                metric="PA",
                wording=fallback_wording,
                target_month=target_month,
                opening_date=opening_date,
                closing_date=closing_date,
                question_type="legacy_pa",
                resolution_source=None,
                is_test_mode=is_test_mode,
            )
        )
        return questions

    if "question_template" in cfg:
        wording = cfg["question_template"].format(
            country_name=country_name,
            start_date=opening_str,
            end_date=closing_str,
        )
        metric = cfg.get("metric_default", "PA")
        resolution_source = cfg.get("resolution_source")
        questions.append(
            _build_question_row(
                hs_run_id=hs_run_id,
                scenario=scenario,
                scenario_ids=[scenario_id],
                iso3=iso3,
                hazard_code=hazard_code,
                hazard_cfg=cfg,
                metric=metric,
                wording=wording,
                target_month=target_month,
                opening_date=opening_date,
                closing_date=closing_date,
                question_type=cfg.get("question_type"),
                resolution_source=resolution_source,
                is_test_mode=is_test_mode,
            )
        )
        return questions

    for question_type in cfg.get("question_types", []):
        template = DISPLACEMENT_TEMPLATES.get(question_type)
        if not template:
            continue

        wording = template.format(
            country_name=country_name,
            start_date=opening_str,
            end_date=closing_str,
        )

        if question_type == "acled_fatalities":
            metric = "FATALITIES"
            resolution_source = "ACLED"
        else:
            metric = "PA"
            resolution_source = "IDMC/DTM"

        questions.append(
            _build_question_row(
                hs_run_id=hs_run_id,
                scenario=scenario,
                scenario_ids=[scenario_id],
                iso3=iso3,
                hazard_code=hazard_code,
                hazard_cfg=cfg,
                metric=metric,
                wording=wording,
                target_month=target_month,
                opening_date=opening_date,
                closing_date=closing_date,
                question_type=question_type,
                resolution_source=resolution_source,
                is_test_mode=is_test_mode,
            )
        )

    return questions


def log_hs_run_to_db(
    hs_run_id: str,
    iso3_list: list[str],
    git_sha: str | None,
    config_profile: str,
) -> None:
    con = connect(read_only=False)
    ensure_schema(con)

    countries_json = json.dumps(sorted(set(iso3_list)))
    now = datetime.utcnow()

    con.execute("DELETE FROM hs_runs WHERE hs_run_id = ?;", [hs_run_id])
    con.execute(
        """
        INSERT INTO hs_runs (hs_run_id, generated_at, git_sha, config_profile, countries_json)
        VALUES (?, ?, ?, ?, ?);
        """,
        [hs_run_id, now, git_sha or "", config_profile, countries_json],
    )
    con.close()


def log_hs_scenarios_to_db(hs_run_id: str, scenarios: Iterable[dict]) -> None:
    con = connect(read_only=False)
    ensure_schema(con)

    for sc in scenarios:
        scenario_id = sc.get("scenario_id") or _stable_hash(
            {
                "iso3": sc.get("iso3"),
                "hazard_code": sc.get("hazard_code"),
                "title": sc.get("scenario_title") or sc.get("title"),
                "hs_run_id": hs_run_id,
            }
        )

        iso3 = (sc.get("iso3") or "").upper()
        hazard_code = (sc.get("hazard_code") or "").upper()
        scenario_title = sc.get("scenario_title") or sc.get("title") or ""
        likely_month = sc.get("likely_month") or sc.get("likely_window_month") or ""
        probability_pct = sc.get("probability_pct")
        pin_best_guess = sc.get("pin_best_guess")
        pa_best_guess = sc.get("pa_best_guess")
        scenario_markdown = sc.get("scenario_markdown") or sc.get("markdown") or ""

        raw_json = sc.get("raw_json")
        if raw_json is None:
            raw_json = sc.get("json")
        if isinstance(raw_json, str):
            scenario_json = raw_json
        else:
            scenario_json = json.dumps(raw_json or sc, ensure_ascii=False)

        con.execute(
            "DELETE FROM hs_scenarios WHERE hs_run_id = ? AND scenario_id = ?;",
            [hs_run_id, scenario_id],
        )
        con.execute(
            """
            INSERT INTO hs_scenarios (
                hs_run_id,
                scenario_id,
                iso3,
                hazard_code,
                scenario_title,
                likely_month,
                probability_pct,
                pin_best_guess,
                pa_best_guess,
                scenario_markdown,
                scenario_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                hs_run_id,
                scenario_id,
                iso3,
                hazard_code,
                scenario_title,
                likely_month,
                probability_pct,
                pin_best_guess,
                pa_best_guess,
                scenario_markdown,
                scenario_json,
            ],
        )

    con.close()


def log_hs_country_reports_to_db(hs_run_id: str, reports: dict[str, dict]) -> None:
    con = connect(read_only=False)
    ensure_schema(con)

    for iso3, payload in reports.items():
        iso3_norm = iso3.upper()
        markdown = payload.get("markdown") or ""
        sources = payload.get("sources") or []
        sources_json = json.dumps(sources, ensure_ascii=False)

        con.execute(
            "DELETE FROM hs_country_reports WHERE hs_run_id = ? AND iso3 = ?;",
            [hs_run_id, iso3_norm],
        )
        con.execute(
            """
            INSERT INTO hs_country_reports (hs_run_id, iso3, report_markdown, sources_json)
            VALUES (?, ?, ?, ?);
            """,
            [hs_run_id, iso3_norm, markdown, sources_json],
        )

    con.close()


def log_hs_questions_to_db(hs_run_id: str, question_rows: Iterable[dict]) -> None:
    con = connect(read_only=False)
    ensure_schema(con)

    for q in question_rows:
        question_id = q["question_id"]
        scenario_ids = q.get("scenario_ids") or []
        scenario_ids_json = json.dumps(scenario_ids, ensure_ascii=False)

        meta = q.get("pythia_metadata") or {}
        meta.setdefault("source", "HS")
        meta.setdefault("purpose", "hs_pipeline")
        pythia_metadata_json = json.dumps(meta, ensure_ascii=False)

        con.execute("DELETE FROM questions WHERE question_id = ?;", [question_id])
        con.execute(
            """
            INSERT INTO questions (
                question_id,
                hs_run_id,
                scenario_ids_json,
                iso3,
                hazard_code,
                metric,
                target_month,
                window_start_date,
                window_end_date,
                wording,
                status,
                pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                question_id,
                hs_run_id,
                scenario_ids_json,
                q.get("iso3"),
                q.get("hazard_code"),
                q.get("metric"),
                q.get("target_month"),
                q.get("window_start_date"),
                q.get("window_end_date"),
                q.get("wording"),
                q.get("status") or "active",
                pythia_metadata_json,
            ],
        )

    con.close()


def upsert_hs_payload(
    hs_run_id: str,
    scenarios: list[dict],
    *,
    today: date,
    horizon_months: int,
    is_test_mode: bool = False,
):
    # Compute common target month and forecast window for this HS run
    target_month = _to_target_month(today, horizon_months)
    opening_date, closing_date = _compute_forecast_window(today, target_month)
    opening_str = opening_date.strftime("%d %B %Y")
    closing_str = closing_date.strftime("%d %B %Y")

    scenario_rows, question_rows = [], []
    for sc in scenarios:
        hz = sc.get("hazard_code", "").strip().upper()
        hz_is_conflict = hz in CONFLICT_HAZARDS or hz.startswith("CONFLICT")
        if hz not in ALLOWED and not hz_is_conflict:
            continue
        iso3 = sc["iso3"].upper()
        s_id = sid(iso3, hz, sc.get("title", ""), sc.get("json", {}))

        raw_json = sc.get("json") or {}
        if is_test_mode and isinstance(raw_json, dict):
            raw_json = {**raw_json, "test_mode": True}

        scenario_payload = {
            "scenario_id": s_id,
            "hs_run_id": hs_run_id,
            "iso3": iso3,
            "country_name": sc.get("country_name", ""),
            "hazard_code": hz,
            "hazard_label": sc.get("hazard_label", ""),
            "likely_month": sc.get("likely_window_month", ""),
            "likely_window_month": sc.get("likely_window_month", ""),
            "markdown": sc.get("markdown", ""),
            "scenario_title": sc.get("scenario_title") or sc.get("title", ""),
            "probability_text": sc.get("probability_text", ""),
            "probability_pct": float(sc.get("probability_pct") or 0.0),
            "pin_best_guess": int(sc.get("pin_best_guess") or 0),
            "pa_best_guess": int(sc.get("pa_best_guess") or 0),
            "best_guess": sc.get("best_guess") or {},
            "json": raw_json,
        }

        scenario_rows.append(scenario_payload)

        question_rows.extend(
            _build_questions_for_scenario(
                hs_run_id=hs_run_id,
                scenario=scenario_payload,
                target_month=target_month,
                opening_date=opening_date,
                closing_date=closing_date,
                opening_str=opening_str,
                closing_str=closing_str,
                is_test_mode=is_test_mode,
            )
        )

    # --- Normalize JSON payloads for DuckDB and add diagnostics ---

    # Log counts before upsert
    logging.info(
        "Horizon Scanner DB payload prepared: %d scenario_rows, %d question_rows for run_id=%s",
        len(scenario_rows),
        len(question_rows),
        hs_run_id,
    )

    # Normalize scenario JSON to valid JSON strings (if not already)
    for row in scenario_rows:
        raw = row.get("json", {})
        if isinstance(raw, str):
            continue
        try:
            row["json"] = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            logging.warning(
                "Failed to JSON-serialize scenario json for scenario_id=%s; falling back to empty object.",
                row.get("scenario_id"),
            )
            row["json"] = "{}"

    # Normalize question hs_json to valid JSON strings (if not already)
    for row in question_rows:
        raw = row.get("hs_json", {})
        if isinstance(raw, str):
            continue
        try:
            row["hs_json"] = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            logging.warning(
                "Failed to JSON-serialize hs_json for question_id=%s; falling back to empty object.",
                row.get("question_id"),
            )
            row["hs_json"] = "{}"

    if scenario_rows:
        sample = scenario_rows[0]
        logging.info(
            "Example hs_scenarios row before upsert | scenario_id=%s | iso3=%s | hazard=%s | prob_pct=%.1f | pa_best_guess=%d",
            sample.get("scenario_id"),
            sample.get("iso3"),
            sample.get("hazard_code"),
            sample.get("probability_pct", 0.0),
            sample.get("pa_best_guess", 0),
        )

    if question_rows:
        sample = question_rows[0]
        logging.info(
            "Example question row before upsert | question_id=%s | iso3=%s | hazard=%s | metric=%s | hs_json_preview=%r",
            sample.get("question_id"),
            sample.get("iso3"),
            sample.get("hazard_code"),
            sample.get("metric"),
            (str(sample.get("hs_json"))[:200] + "...")
            if sample.get("hs_json") is not None
            else None,
        )

    if scenario_rows:
        log_hs_scenarios_to_db(hs_run_id, scenario_rows)
    if question_rows:
        log_hs_questions_to_db(hs_run_id, question_rows)

    return scenario_rows, question_rows
