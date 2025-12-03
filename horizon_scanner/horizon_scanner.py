from __future__ import annotations

"""Horizon Scanner v2 triage entrypoint.

This version replaces the previous risk-report generator with a
triage-first pipeline that writes structured outputs into the ``hs_triage``
table and runs with bounded concurrency.
"""

import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from forecaster.providers import GEMINI_MODEL_ID, ModelSpec, call_chat_ms, estimate_cost_usd
from horizon_scanner.db_writer import BLOCKED_HAZARDS, HAZARD_CONFIG
from horizon_scanner.prompts import build_hs_triage_prompt
from pythia.db.schema import connect as pythia_connect, ensure_schema

# Ensure package imports resolve when executed as a script
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HS_MAX_WORKERS = int(os.getenv("HS_MAX_WORKERS", "6"))
HS_TEMPERATURE = float(os.getenv("HS_TEMPERATURE", "0.3"))
COUNTRIES_CSV = REPO_ROOT / "resolver" / "data" / "countries.csv"
EMDAT_SHOCK_MAP = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical_cyclone",
    "HW": "heat_wave",
}


def _resolve_hs_model() -> str:
    model_id = (GEMINI_MODEL_ID or "").strip()
    if model_id:
        return model_id
    return os.getenv("HS_MODEL_ID", "gemini-2.5-flash-lite")


def _load_country_registry() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return lookup maps for iso3->name and name_norm->iso3."""

    if not COUNTRIES_CSV.exists():
        return {}, {}

    iso3_to_name: Dict[str, str] = {}
    name_to_iso3: Dict[str, str] = {}
    try:
        with open(COUNTRIES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("country_name") or "").strip()
                iso3 = (row.get("iso3") or "").strip().upper()
                if not name or not iso3:
                    continue
                iso3_to_name[iso3] = name
                name_to_iso3[name.lower()] = iso3
    except Exception:
        logger.exception("Failed to read country registry from %s", COUNTRIES_CSV)
    return iso3_to_name, name_to_iso3


def _resolve_country(raw: str, iso3_to_name: Dict[str, str], name_to_iso3: Dict[str, str]) -> Tuple[str, str]:
    candidate = (raw or "").strip()
    if not candidate:
        return "", ""

    upper = candidate.upper()
    if len(upper) == 3 and upper in iso3_to_name:
        return iso3_to_name[upper], upper

    lower = candidate.lower()
    if lower in name_to_iso3:
        iso3 = name_to_iso3[lower]
        return iso3_to_name.get(iso3, candidate), iso3

    return candidate, upper[:3]


def _build_hazard_catalog() -> Dict[str, str]:
    catalog: Dict[str, str] = {}
    for code, cfg in HAZARD_CONFIG.items():
        if code in BLOCKED_HAZARDS:
            continue
        label = cfg.get("label") or code
        catalog[code] = label
    return dict(sorted(catalog.items()))


def _build_resolver_features_for_country(iso3: str) -> Dict[str, Any]:
    """Summarize Resolver history per hazard to ground triage."""

    con = pythia_connect(read_only=True)

    def _summarize(values: list[float], source: str, default_note: str, short_ok: bool = True) -> Dict[str, Any]:
        if not values:
            return {
                "source": source,
                "history_length": 0,
                "recent_mean": None,
                "recent_max": None,
                "recent_trend": "uncertain",
                "data_quality": "low",
                "notes": default_note,
            }
        history_length = len(values)
        last_6 = values[-6:]
        trend = "uncertain"
        if len(last_6) >= 2:
            if last_6[-1] > last_6[0]:
                trend = "up"
            elif last_6[-1] < last_6[0]:
                trend = "down"
            else:
                trend = "flat"
        quality = "high" if history_length >= 24 else "medium" if short_ok else "low"
        return {
            "source": source,
            "history_length": history_length,
            "recent_mean": float(sum(last_6) / len(last_6)) if last_6 else None,
            "recent_max": float(max(last_6)) if last_6 else None,
            "recent_trend": trend,
            "data_quality": quality,
            "notes": default_note,
        }

    try:
        conflict_rows = con.execute(
            """
            SELECT fatalities
            FROM acled_monthly_fatalities
            WHERE iso3 = ?
            ORDER BY ym
            """,
            [iso3],
        ).fetchall()
        conflict_values = [r[0] for r in conflict_rows if r and r[0] is not None]
        conflict_summary = _summarize(
            conflict_values,
            "ACLED",
            "ACLED coverage is relatively strong for conflict fatalities.",
        )

        idmc_rows = con.execute(
            """
            SELECT value
            FROM facts_deltas
            WHERE iso3 = ? AND metric = 'idp_displacement_stock_idmc'
            ORDER BY ym
            """,
            [iso3],
        ).fetchall()
        idmc_values = [r[0] for r in idmc_rows if r and r[0] is not None]
        idmc_summary = _summarize(
            idmc_values,
            "IDMC",
            "IDMC history is short; treat as a weak prior.",
            short_ok=bool(len(idmc_values) >= 12),
        )

        emdat_summaries: Dict[str, Dict[str, Any]] = {}
        for hz, shock in EMDAT_SHOCK_MAP.items():
            rows = con.execute(
                """
                SELECT pa
                FROM emdat_pa
                WHERE iso3 = ? AND shock_type = ?
                ORDER BY ym
                """,
                [iso3, shock],
            ).fetchall()
            values = [r[0] for r in rows if r and r[0] is not None]
            emdat_summaries[hz] = _summarize(
                values,
                "EM-DAT" if values else "none",
                "EM-DAT coverage can be patchy; smaller events may be missing." if values else "No reliable EM-DAT series for this hazard.",
                short_ok=bool(values),
            )

        conflict_codes = {"ACO", "ACE", "CU", "CONFLICT", "POLITICAL_VIOLENCE", "CIVIL_CONFLICT", "URBAN_CONFLICT"}
        features: Dict[str, Any] = {}
        for code, cfg in HAZARD_CONFIG.items():
            if code in BLOCKED_HAZARDS:
                continue
            up = code.upper()
            if up in conflict_codes:
                features[code] = conflict_summary | {"metric": "fatalities"}
            elif up == "DI":
                features[code] = {
                    "source": "none",
                    "history_length": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "recent_trend": "uncertain",
                    "data_quality": "low",
                    "notes": "No resolver base rate for displacement inflow; rely on HS + research.",
                }
            elif up in {"FL", "DR", "TC", "HW"}:
                features[code] = emdat_summaries.get(up, {
                    "source": "none",
                    "history_length": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "recent_trend": "uncertain",
                    "data_quality": "low",
                    "notes": "No EM-DAT series available.",
                }) | {"metric": "affected"}
            else:
                features[code] = idmc_summary | {"metric": "displacement"}

        return dict(sorted(features.items()))
    finally:
        con.close()


def _write_hs_triage(run_id: str, iso3: str, triage: Dict[str, Any]) -> None:
    con = pythia_connect(read_only=False)
    try:
        hazards = triage.get("hazards") or {}
        for hz_code, hdata in hazards.items():
            tier = (hdata.get("tier") or "quiet").lower()
            score = float(hdata.get("triage_score") or 0.0)

            need_full_spd = False
            if tier == "priority" or score >= 0.7:
                need_full_spd = True
            elif tier == "watchlist" and 0.4 <= score < 0.7:
                need_full_spd = True

            con.execute(
                """
                INSERT INTO hs_triage (
                    run_id, iso3, hazard_code,
                    tier, triage_score, need_full_spd,
                    drivers_json, regime_shifts_json, data_quality_json, scenario_stub
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    iso3,
                    hz_code,
                    tier,
                    score,
                    need_full_spd,
                    json.dumps(hdata.get("drivers") or []),
                    json.dumps(hdata.get("regime_shifts") or []),
                    json.dumps(hdata.get("data_quality") or {}),
                    hdata.get("scenario_stub") or "",
                ],
            )
    finally:
        con.close()


async def _call_hs_model(prompt_text: str) -> tuple[str, Dict[str, Any], str]:
    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=_resolve_hs_model(),
        active=True,
    )
    return await call_chat_ms(
        spec,
        prompt_text,
        temperature=HS_TEMPERATURE,
        prompt_key="hs.triage.v2",
        prompt_version="1.0.0",
        component="HorizonScanner",
    )


def _run_hs_for_country(run_id: str, iso3: str, country_name: str) -> None:
    logger.info("Running HS triage for %s (%s)", country_name, iso3)
    hazard_catalog = _build_hazard_catalog()
    resolver_features = _build_resolver_features_for_country(iso3)

    prompt = build_hs_triage_prompt(
        country_name=country_name,
        iso3=iso3,
        hazard_catalog=hazard_catalog,
        resolver_features=resolver_features,
        model_info={},
    )

    text, usage, error = asyncio.run(_call_hs_model(prompt))
    if error:
        raise RuntimeError(error)

    triage = json.loads(text)
    _write_hs_triage(run_id, iso3, triage)

    cost = usage.get("cost_usd") or estimate_cost_usd(_resolve_hs_model(), usage)
    logger.info(
        "HS triage completed for %s (%s): tokens=%s cost_usd=%.4f",
        country_name,
        iso3,
        usage.get("total_tokens"),
        cost or 0.0,
    )


def _load_country_list(provided: list[str] | None) -> list[Tuple[str, str]]:
    iso3_to_name, name_to_iso3 = _load_country_registry()

    if provided:
        raw_countries = provided
    else:
        country_list_path = CURRENT_DIR / "hs_country_list.txt"
        with open(country_list_path, "r", encoding="utf-8") as f:
            raw_countries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    resolved: list[Tuple[str, str]] = []
    for raw in raw_countries:
        name, iso3 = _resolve_country(raw, iso3_to_name, name_to_iso3)
        if not iso3:
            logger.warning("Skipping empty country entry for %r", raw)
            continue
        resolved.append((name, iso3))
    return resolved


def main(countries: list[str] | None = None):
    logger.info("Starting Horizon Scanner triage run...")
    ensure_schema()

    start_time = datetime.utcnow()
    run_id = f"hs_{start_time.strftime('%Y%m%dT%H%M%S')}"
    os.environ["PYTHIA_HS_RUN_ID"] = run_id

    country_entries = _load_country_list(countries)
    if not country_entries:
        logger.warning("No countries supplied to Horizon Scanner; exiting.")
        return

    logger.info("Processing %d countries with max %d workers", len(country_entries), HS_MAX_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=HS_MAX_WORKERS) as pool:
        futures = [
            pool.submit(_run_hs_for_country, run_id, iso3, name)
            for name, iso3 in country_entries
        ]

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                logger.exception("HS triage failed for one country")

    logger.info("Horizon Scanner triage run complete for %d countries", len(country_entries))


if __name__ == "__main__":
    main()
