# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

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
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from forecaster.providers import GEMINI_MODEL_ID, ModelSpec, call_chat_ms, estimate_cost_usd
from horizon_scanner.db_writer import (
    BLOCKED_HAZARDS,
    HAZARD_CONFIG,
    log_hs_run_to_db,
)
from horizon_scanner.prompts import build_hs_triage_prompt
from horizon_scanner.llm_logging import log_hs_llm_call
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
            logger.debug("Skipping blocked hazard %s in HS triage", code)
            continue
        label = cfg.get("label") or code
        catalog[code] = label
    return dict(sorted(catalog.items()))


def _build_resolver_features_for_country(iso3: str) -> Dict[str, Any]:
    """Summarize Resolver history per hazard to ground triage."""

    con = pythia_connect(read_only=True)
    features: Dict[str, Any] = {"iso3": iso3}

    # Conflict/fatalities base-rate features
    try:
        rows = con.execute(
            """
            SELECT month, fatalities
            FROM acled_monthly_fatalities
            WHERE iso3 = ?
            ORDER BY month
            """,
            [iso3],
        ).fetchall()
        values = [int(r[1]) for r in rows]
        n = len(values)
        if n:
            recent = values[-min(6, n) :]
            features["conflict"] = {
                "source": "ACLED",
                "history_length": n,
                "recent_mean": sum(recent) / len(recent),
                "recent_max": max(recent),
                "trend": "up"
                if n >= 2 and recent[-1] > recent[0]
                else "down"
                if n >= 2 and recent[-1] < recent[0]
                else "flat",
                "data_quality": "high",
                "notes": "ACLED coverage is relatively strong.",
            }
        else:
            features["conflict"] = {
                "source": "ACLED",
                "history_length": 0,
                "data_quality": "low",
                "notes": "No ACLED history for this country.",
            }
    except Exception as exc:  # pragma: no cover - defensive fallback for schema drift
        logging.warning("Resolver conflict features failed for %s: %s", iso3, exc)
        features["conflict"] = {
            "source": "ACLED",
            "history_length": 0,
            "data_quality": "unknown",
            "notes": f"Resolver conflict features unavailable: {type(exc).__name__}",
        }

    # TODO: similar minimalist blocks for IDMC, EM-DAT if you want them
    # For now, return at least the conflict bundle, plus a stub:
    features.setdefault("displacement", {"notes": "Not implemented"})
    features.setdefault("natural_hazards", {"notes": "Not implemented"})

    con.close()
    return features


def _write_hs_triage(run_id: str, iso3: str, triage: Dict[str, Any]) -> None:
    con = pythia_connect(read_only=False)
    try:
        hazards = triage.get("hazards") or {}
        for hz_code, hdata in hazards.items():
            tier = (hdata.get("tier") or "quiet").lower()
            score = float(hdata.get("triage_score") or 0.0)

            hz_up = (hz_code or "").upper().strip()
            iso3_up = (iso3 or "").upper().strip()

            # Retire ACO in favour of ACE as the canonical conflict hazard code.
            if hz_up == "ACO":
                logger.info(
                    "HS triage: skipping ACO hazard for %s; ACE is the canonical conflict hazard.",
                    iso3_up,
                )
                continue

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
                    iso3_up,
                    hz_up,
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


async def _call_hs_model(prompt_text: str) -> tuple[str, Dict[str, Any], str, ModelSpec]:
    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=_resolve_hs_model(),
        active=True,
    )
    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            spec,
            prompt_text,
            temperature=HS_TEMPERATURE,
            prompt_key="hs.triage.v2",
            prompt_version="1.0.0",
            component="HorizonScanner",
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", spec

    usage = usage or {}
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    return text, usage, error, spec


def _run_hs_for_country(run_id: str, iso3: str, country_name: str) -> None:
    con = pythia_connect(read_only=False)
    try:
        iso3_up = (iso3 or "").upper()
        logger.info("Running HS triage for %s (%s)", country_name, iso3_up)
        resolver_features = _build_resolver_features_for_country(iso3)
        hazard_catalog = _build_hazard_catalog()
        prompt = build_hs_triage_prompt(
            country_name=country_name,
            iso3=iso3_up,
            hazard_catalog=hazard_catalog,
            resolver_features=resolver_features,
            model_info={},
        )

        call_start = time.time()
        text, usage, error, model_spec = asyncio.run(_call_hs_model(prompt))
        usage = usage or {}
        usage.setdefault("elapsed_ms", int((time.time() - call_start) * 1000))

        log_error_text: str | None = None
        triage: Dict[str, Any] = {}
        hazards_dict: Dict[str, Any] = {}

        if error:
            log_error_text = str(error)
        else:
            raw = (text or "").strip()
            if not raw:
                log_error_text = "empty response"
            else:
                fence_match = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.S)
                if not fence_match:
                    debug_dir = Path("debug/hs_triage_raw")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = debug_dir / f"{run_id}__{iso3}.txt"
                    raw_path.write_text(text or "", encoding="utf-8")
                    log_error_text = "missing JSON fences"
                    logger.error(
                        "HS triage response missing JSON fences for %s; raw saved to %s",
                        iso3_up,
                        raw_path,
                    )
                else:
                    payload = fence_match.group(1).strip()
                    try:
                        triage = json.loads(payload)
                        hazards_dict = triage.get("hazards") or {}
                    except json.JSONDecodeError as exc:
                        debug_dir = Path("debug/hs_triage_raw")
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        raw_path = debug_dir / f"{run_id}__{iso3}.txt"
                        raw_path.write_text(text or "", encoding="utf-8")
                        log_error_text = f"json decode failed: {exc}"
                        logger.error(
                            "HS triage JSON decode failed for %s: %s (raw saved to %s)",
                            iso3_up,
                            exc,
                            raw_path,
                        )

        hazards_dict = hazards_dict if isinstance(hazards_dict, dict) else {}
        if hazards_dict:
            for hz_code in hazards_dict.keys():
                hz_code_up = (hz_code or "").upper().strip()
                log_hs_llm_call(
                    hs_run_id=run_id,
                    iso3=iso3_up,
                    hazard_code=hz_code_up,
                    model_spec=model_spec,
                    prompt_text=prompt,
                    response_text=text or "",
                    usage=usage,
                    error_text=log_error_text,
                )
                logger.info(
                    "HS logged triage call: hs_run_id=%s iso3=%s hazard=%s",
                    run_id,
                    iso3_up,
                    hz_code_up,
                )
        else:
            log_hs_llm_call(
                hs_run_id=run_id,
                iso3=iso3_up,
                hazard_code="",
                model_spec=model_spec,
                prompt_text=prompt,
                response_text=text or "",
                usage=usage,
                error_text=log_error_text,
            )
            logger.info(
                "HS logged triage call: hs_run_id=%s iso3=%s hazard=%s (no hazards in triage JSON)",
                run_id,
                iso3_up,
                "",
            )

        if log_error_text:
            logger.error("HS triage error for %s: %s", iso3_up, log_error_text)
            return

        _write_hs_triage(run_id, iso3_up, triage)

        cost = usage.get("cost_usd") or estimate_cost_usd(_resolve_hs_model(), usage)
        logger.info(
            "HS triage completed for %s (%s): tokens=%s cost_usd=%.4f",
            country_name,
            iso3,
            usage.get("total_tokens"),
            cost or 0.0,
        )
    finally:
        con.close()


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

    iso3_list = [iso3 for (_name, iso3) in country_entries]
    try:
        git_sha = os.getenv("GITHUB_SHA") or ""
        config_profile = os.getenv("PYTHIA_CONFIG_PROFILE", "default")
        log_hs_run_to_db(run_id, iso3_list, git_sha=git_sha, config_profile=config_profile)
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("Failed to log hs_run %s: %s", run_id, exc)

    logger.info("Horizon Scanner triage run complete for %d countries", len(country_entries))


if __name__ == "__main__":
    main()
