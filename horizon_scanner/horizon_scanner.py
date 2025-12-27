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
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict

from forecaster.providers import (
    GEMINI_MODEL_ID,
    ModelSpec,
    call_chat_ms,
    estimate_cost_usd,
    reset_provider_failures_for_run,
)
from horizon_scanner.db_writer import (
    BLOCKED_HAZARDS,
    HAZARD_CONFIG,
    get_expected_hs_hazards,
    log_hs_country_reports_to_db,
    log_hs_run_to_db,
)
from horizon_scanner.prompts import build_hs_triage_prompt
from horizon_scanner.llm_logging import log_hs_llm_call
from pythia.db.schema import connect as pythia_connect, ensure_schema
from pythia.web_research import fetch_evidence_pack

# Ensure package imports resolve when executed as a script
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HS_MAX_WORKERS = int(os.getenv("HS_MAX_WORKERS", "6"))
HS_TEMPERATURE = float(os.getenv("HS_TEMPERATURE", "0.0"))
HS_PRIORITY_THRESHOLD = float(os.getenv("PYTHIA_HS_PRIORITY_THRESHOLD", "0.70"))
HS_WATCHLIST_THRESHOLD = float(os.getenv("PYTHIA_HS_WATCHLIST_THRESHOLD", "0.40"))
HS_HYSTERESIS_BAND = float(os.getenv("PYTHIA_HS_HYSTERESIS_BAND", "0.05"))
HS_EVIDENCE_MAX_SOURCES = int(os.getenv("PYTHIA_HS_EVIDENCE_MAX_SOURCES", "8"))
HS_EVIDENCE_MAX_SIGNALS = int(os.getenv("PYTHIA_HS_EVIDENCE_MAX_SIGNALS", "8"))
COUNTRIES_CSV = REPO_ROOT / "resolver" / "data" / "countries.csv"
EMDAT_SHOCK_MAP = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical_cyclone",
    "HW": "heat_wave",
}
_COUNTRY_ALIASES = {
    "democratic republic of congo": "COD",
    "democratic republic of the congo": "COD",
    "dr congo": "COD",
    "drc": "COD",
    "congo (drc)": "COD",
    "congo, dem. rep.": "COD",
    "cote d'ivoire": "CIV",
    "cote divoire": "CIV",
    "cote d’ivoire": "CIV",
}


def _norm_country_key(raw: str) -> str:
    """Normalise a country key for robust matching.

    - Lowercase
    - Strip diacritics
    - Replace non-alphanumeric characters with spaces
    - Collapse whitespace
    """

    if raw is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(raw))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _build_hs_evidence_query(country_name: str, iso3: str) -> str:
    """Build a grounded web-research query for HS triage."""

    iso3_val = (iso3 or "").strip().upper()
    name = (country_name or "").strip()
    label = f"{name} ({iso3_val})" if name else iso3_val
    return (
        f"{label} humanitarian risk outlook - fetch grounded recent signals (last 120 days) "
        "across conflict, displacement, disasters, food security, and political stability. "
        "Also include concise structural drivers (max 8 lines) as background context."
    )


def _render_evidence_markdown(pack: dict[str, Any]) -> str:
    structural = (pack.get("structural_context") or "(none)").strip()
    recent_signals = pack.get("recent_signals") or []
    sources = pack.get("sources") or []

    lines = ["# Evidence pack", ""]
    lines.append(f"Query: {pack.get('query', '')}")
    if pack.get("recency_days"):
        lines.append(f"Recency window: last {pack.get('recency_days')} days")
    lines.append(f"Grounded: {bool(pack.get('grounded'))}")
    lines.append("")
    lines.append("Structural context (background only):")
    lines.append(structural if structural else "(none)")
    lines.append("")
    lines.append("Recent signals (last window):")
    if recent_signals:
        for sig in recent_signals:
            lines.append(f"- {sig}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Sources:")
    if sources:
        for src in sources:
            title = src.get("title") or src.get("url") or "(untitled)"
            url = src.get("url") or ""
            lines.append(f"- {title} — {url}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _sort_sources_by_url(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _url_key(src: dict[str, Any]) -> str:
        return str(src.get("url") or "").strip()

    return sorted(sources, key=_url_key)


def _maybe_build_country_evidence_pack(run_id: str, iso3: str, country_name: str) -> dict[str, Any] | None:
    retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    if not retriever_enabled and os.getenv("PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED", "0") != "1":
        return None

    pack: dict[str, Any] | None = None
    try:
        query = _build_hs_evidence_query(country_name, iso3)
        model_id = (os.getenv("PYTHIA_RETRIEVER_MODEL_ID") or "").strip() if retriever_enabled else None
        pack = dict(
            fetch_evidence_pack(
                query,
                purpose="hs_country_pack",
                run_id=run_id,
                hs_run_id=run_id,
                model_id=model_id or None,
            )
            or {}
        )
    except Exception as exc:  # noqa: BLE001 - defensive around web research
        logger.warning("HS web research failed for %s: %s", iso3, exc)
        pack = {
            "query": query if "query" in locals() else iso3,
            "recency_days": 120,
            "grounded": False,
            "sources": [],
            "structural_context": "",
            "recent_signals": [],
            "debug": {"error": f"{exc}"},
        }

    pack = dict(pack or {})
    sources_raw = pack.get("sources") or []
    sources_list = [src for src in sources_raw if isinstance(src, dict)]
    signals_raw = pack.get("recent_signals") or []
    signals_list = [str(sig) for sig in signals_raw if str(sig).strip()]

    sources_before = len(sources_list)
    signals_before = len(signals_list)
    sources_list = _sort_sources_by_url(sources_list)[:HS_EVIDENCE_MAX_SOURCES]
    signals_list = signals_list[:HS_EVIDENCE_MAX_SIGNALS]
    pack["sources"] = sources_list
    pack["recent_signals"] = signals_list

    markdown = _render_evidence_markdown(pack)
    debug = pack.get("debug") or {}
    grounding_debug = {
        "groundingSupports_count": debug.get("groundingSupports_count", 0),
        "groundingChunks_count": debug.get("groundingChunks_count", 0),
        "webSearchQueries_len": len(debug.get("webSearchQueries") or []),
        "n_sources_before": sources_before,
        "n_sources_after": len(sources_list),
        "n_signals_before": signals_before,
        "n_signals_after": len(signals_list),
        "reason_code": debug.get("reason_code"),
    }
    pack["grounding_debug"] = grounding_debug
    pack["grounded"] = bool(pack.get("grounded"))
    pack.setdefault("structural_context", "")
    pack.setdefault("recent_signals", [])

    try:
        log_hs_country_reports_to_db(
            run_id,
            {
                iso3.upper(): {
                    "markdown": markdown,
                    "sources": pack.get("sources") or [],
                    "grounded": pack.get("grounded", False),
                    "grounding_debug": grounding_debug,
                    "structural_context": pack.get("structural_context") or "",
                    "recent_signals": pack.get("recent_signals") or [],
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist HS evidence pack for %s: %s", iso3, exc)

    pack["markdown"] = markdown
    return pack


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
                name = (row.get("country_name") or row.get("\ufeffcountry_name") or "").strip()
                iso3 = (row.get("iso3") or "").strip().upper()
                if not name or not iso3:
                    continue
                iso3_to_name[iso3] = name
                name_to_iso3[_norm_country_key(name)] = iso3
    except Exception:
        logger.exception("Failed to read country registry from %s", COUNTRIES_CSV)

    for alias, iso3 in _COUNTRY_ALIASES.items():
        if iso3 in iso3_to_name:
            name_to_iso3.setdefault(_norm_country_key(alias), iso3)
        else:
            logger.warning("Skipping country alias %r -> %s (iso3 not in registry)", alias, iso3)
    return iso3_to_name, name_to_iso3

def _resolve_country(
    candidate: str, iso3_to_name: Dict[str, str], name_to_iso3: Dict[str, str]
) -> tuple[str, str | None]:
    raw = (candidate or "").strip()
    if not raw:
        return candidate, None

    normalized = _norm_country_key(raw)

    if normalized in name_to_iso3:
        iso3 = name_to_iso3[normalized]
        return iso3_to_name.get(iso3, raw), iso3

    if len(raw) == 3 and raw.isalpha():
        iso3 = raw.upper()
        if iso3 in iso3_to_name:
            return iso3_to_name[iso3], iso3
        return raw, None

    return raw, None


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

    def _trend_from(values: list[float]) -> str:
        if len(values) >= 2:
            if values[-1] > values[0]:
                return "up"
            if values[-1] < values[0]:
                return "down"
        return "flat"

    def _last_values(series: list[tuple[str, float]], limit: int = 6) -> list[dict[str, float]]:
        tail = series[-limit:]
        return [{"ym": ym, "value": val} for ym, val in tail]

    def _coerce_ym(raw_ym: Any) -> str:
        try:
            if hasattr(raw_ym, "isoformat"):
                return raw_ym.isoformat()
        except Exception:
            pass
        return str(raw_ym)

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
    # Displacement flows (IDMC/DTM, facts_deltas)
    try:
        displacement_rows = con.execute(
            """
            SELECT ym, SUM(
                CASE
                    WHEN lower(metric) = 'new_displacements' THEN COALESCE(value_new, 0)
                    WHEN lower(metric) = 'idp_displacement_new_dtm' THEN COALESCE(value_new, 0)
                    WHEN lower(metric) = 'idp_displacement_flow_idmc' THEN COALESCE(value_new, 0)
                    ELSE 0
                END
            ) AS flow_value
            FROM facts_deltas
            WHERE upper(iso3) = ?
              AND lower(series_semantics) = 'new'
              AND lower(metric) IN (
                'new_displacements',
                'idp_displacement_new_dtm',
                'idp_displacement_flow_idmc'
              )
            GROUP BY ym
            ORDER BY ym
            """,
            [iso3],
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Resolver displacement features failed for %s: %s", iso3, exc)
        displacement_rows = None

    if displacement_rows:
        disp_series: list[tuple[str, float]] = []
        for ym_val, flow_val in displacement_rows:
            try:
                disp_series.append((_coerce_ym(ym_val), float(flow_val or 0.0)))
            except Exception:
                continue
        values_only = [v for _ym, v in disp_series]
        features["displacement"] = {
            "source": "IDMC/DTM",
            "history_length": len(disp_series),
            "recent_mean": sum(values_only[-6:]) / len(values_only[-6:]) if values_only[-6:] else None,
            "recent_max": max(values_only) if values_only else None,
            "trend": _trend_from(values_only),
            "last_6_values": _last_values(disp_series),
            "data_quality": "medium",
            "notes": "IDMC/DTM monthly displacement flows from facts_deltas (noisy, often sparse).",
        }
    else:
        features["displacement"] = {
            "source": "IDMC/DTM",
            "history_length": 0,
            "data_quality": "low",
            "notes": "No IDMC/DTM flows in DB for this country (common).",
        }

    # Natural hazards (EM-DAT PA)
    features["natural_hazards"] = {}
    for hz_code, shock_type in EMDAT_SHOCK_MAP.items():
        try:
            emdat_rows = con.execute(
                """
                SELECT ym, pa
                FROM emdat_pa
                WHERE upper(iso3) = ? AND lower(shock_type) = ?
                ORDER BY ym
                """,
                [iso3, shock_type],
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Resolver EM-DAT features failed for %s/%s: %s",
                iso3,
                hz_code,
                exc,
            )
            emdat_rows = None

        if emdat_rows:
            nh_series: list[tuple[str, float]] = []
            for ym_val, pa_val in emdat_rows:
                try:
                    nh_series.append((_coerce_ym(ym_val), float(pa_val or 0.0)))
                except Exception:
                    continue
            values_only = [v for _ym, v in nh_series]
            features["natural_hazards"][hz_code] = {
                "source": "EM-DAT",
                "history_length": len(nh_series),
                "recent_mean": sum(values_only[-6:]) / len(values_only[-6:]) if values_only[-6:] else None,
                "recent_max": max(values_only) if values_only else None,
                "trend": _trend_from(values_only),
                "last_6_values": _last_values(nh_series),
                "data_quality": "medium",
                "notes": "EM-DAT PA history (often sparse or missing for some hazards).",
            }
        else:
            features["natural_hazards"][hz_code] = {
                "source": "EM-DAT",
                "history_length": 0,
                "data_quality": "low",
                "notes": "No EM-DAT PA history for this hazard/country (common).",
            }

    con.close()
    return features


def _parse_hs_triage_json(raw: str) -> dict[str, Any]:
    """Parse HS triage output, preferring unfenced JSON but allowing fences."""

    s = (raw or "").strip()
    if not s:
        raise ValueError("empty response")

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced_json = re.search(r"```json\s*(.*?)\s*```", s, flags=re.S | re.I)
    if fenced_json:
        return json.loads(fenced_json.group(1).strip())

    fenced_any = re.search(r"```\s*(.*?)\s*```", s, flags=re.S)
    if fenced_any:
        return json.loads(fenced_any.group(1).strip())

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])

    raise json.JSONDecodeError("could not locate JSON object", s, 0)


def _coerce_score(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _coerce_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _short_error(raw: str | None, limit: int = 200) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _merge_unique(values_a: list[str], values_b: list[str], limit: int = 6) -> list[str]:
    merged = sorted({value for value in values_a + values_b if value})
    return merged[:limit]


def _build_pass_hazards(
    hazards_raw: Dict[str, Any],
    expected_hazards: list[str],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Any] = {}
    if isinstance(hazards_raw, dict):
        for hz_code, hdata in hazards_raw.items():
            key = (hz_code or "").upper().strip()
            if key:
                normalized[key] = hdata if isinstance(hdata, dict) else {}

    pass_hazards: Dict[str, Dict[str, Any]] = {}
    for hz_code in expected_hazards:
        hdata = normalized.get(hz_code) if isinstance(normalized.get(hz_code), dict) else None
        if not hdata:
            pass_hazards[hz_code] = {
                "score": 0.0,
                "drivers": [],
                "regime_shifts": [],
                "data_quality": {"status": "missing_in_model_output"},
                "scenario_stub": "",
            }
            continue

        data_quality = hdata.get("data_quality") or {}
        if not isinstance(data_quality, dict):
            data_quality = {"raw": data_quality}

        pass_hazards[hz_code] = {
            "score": _coerce_score(hdata.get("triage_score")),
            "drivers": _coerce_list(hdata.get("drivers")),
            "regime_shifts": _coerce_list(hdata.get("regime_shifts")),
            "data_quality": data_quality,
            "scenario_stub": str(hdata.get("scenario_stub") or ""),
        }

    return pass_hazards


def tier_from_score(score: float) -> str:
    if score >= HS_PRIORITY_THRESHOLD + HS_HYSTERESIS_BAND:
        return "priority"
    if score >= HS_PRIORITY_THRESHOLD - HS_HYSTERESIS_BAND:
        return "watchlist"
    if score >= HS_WATCHLIST_THRESHOLD + HS_HYSTERESIS_BAND:
        return "watchlist"
    if score >= HS_WATCHLIST_THRESHOLD - HS_HYSTERESIS_BAND:
        return "quiet"
    return "quiet"


def _write_hs_triage(run_id: str, iso3: str, triage: Dict[str, Any], error_text: str | None = None) -> None:
    allowed_hazards = set(_build_hazard_catalog().keys())
    expected_hazards = get_expected_hs_hazards()
    con = pythia_connect(read_only=False)
    try:
        hazards = triage.get("hazards") or {}
        normalized_hazards: Dict[str, Any] = {}
        if isinstance(hazards, dict):
            for hz_code, hdata in hazards.items():
                key = (hz_code or "").upper().strip()
                if key:
                    normalized_hazards[key] = hdata

        iso3_up = (iso3 or "").upper().strip()
        for hz_code in expected_hazards:
            hz_up = (hz_code or "").upper().strip()
            if hz_up == "ACO":
                logger.info(
                    "HS triage: skipping ACO hazard for %s; ACE is the canonical conflict hazard.",
                    iso3_up,
                )
                continue
            if hz_up not in allowed_hazards:
                logger.info(
                    "HS triage: ignoring unknown hazard code from model | run_id=%s iso3=%s hazard=%s",
                    run_id,
                    iso3_up,
                    hz_up,
                )
                continue

            hdata = normalized_hazards.get(hz_up)
            if not isinstance(hdata, dict):
                hdata = {}

            placeholder_reason = ""
            if error_text:
                placeholder_reason = error_text
            elif not hdata:
                placeholder_reason = "missing hazard in triage output"

            if placeholder_reason:
                score = 0.0
                drivers = []
                regime_shifts = []
                data_quality = {"status": "error", "error_text": placeholder_reason}
                scenario_stub = ""
            else:
                score = float(hdata.get("triage_score") or 0.0)
                drivers = hdata.get("drivers") or []
                regime_shifts = hdata.get("regime_shifts") or []
                data_quality = hdata.get("data_quality") or {}
                if not isinstance(data_quality, dict):
                    data_quality = {"raw": data_quality}
                scenario_stub = hdata.get("scenario_stub") or ""

            tier = tier_from_score(score)
            need_full_spd = tier in {"priority", "watchlist"}

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
                    json.dumps(drivers),
                    json.dumps(regime_shifts),
                    json.dumps(data_quality),
                    scenario_stub,
                ],
            )
    finally:
        con.close()


class _TriageCallResult(TypedDict):
    iso3: str
    error_text: str | None
    response_text: str


async def _call_hs_model(
    prompt_text: str, *, run_id: str | None = None
) -> tuple[str, Dict[str, Any], str, ModelSpec]:
    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=_resolve_hs_model(),
        active=True,
        purpose="hs_triage",
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
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", spec

    usage = usage or {}
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    return text, usage, error, spec


def _run_hs_for_country(run_id: str, iso3: str, country_name: str) -> _TriageCallResult:
    con = pythia_connect(read_only=False)
    try:
        iso3_up = (iso3 or "").upper()
        logger.info("Running HS triage for %s (%s)", country_name, iso3_up)
        resolver_features = _build_resolver_features_for_country(iso3)
        hazard_catalog = _build_hazard_catalog()
        evidence_pack = _maybe_build_country_evidence_pack(run_id, iso3_up, country_name)
        prompt = build_hs_triage_prompt(
            country_name=country_name,
            iso3=iso3_up,
            hazard_catalog=hazard_catalog,
            resolver_features=resolver_features,
            model_info={},
            evidence_pack=evidence_pack,
        )

        pass_results: list[dict[str, Any]] = []
        for pass_idx in (1, 2):
            call_start = time.time()
            text, usage, error, model_spec = asyncio.run(_call_hs_model(prompt, run_id=run_id))
            usage = usage or {}
            usage.setdefault("elapsed_ms", int((time.time() - call_start) * 1000))

            log_error_text: str | None = None
            triage: Dict[str, Any] = {}
            hazards_dict: Dict[str, Any] = {}
            parse_ok = True

            if error:
                log_error_text = str(error)
                parse_ok = False
            else:
                raw = (text or "").strip()
                if not raw:
                    log_error_text = "empty response"
                    parse_ok = False
                else:
                    try:
                        triage = _parse_hs_triage_json(raw)
                        hazards_dict = triage.get("hazards") or {}
                    except Exception as exc:  # noqa: BLE001
                        debug_dir = Path("debug/hs_triage_raw")
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        raw_path = debug_dir / f"{run_id}__{iso3}__pass{pass_idx}.txt"
                        raw_path.write_text(text or "", encoding="utf-8")
                        log_error_text = f"triage parse failed: {type(exc).__name__}: {exc}"
                        parse_ok = False
                        logger.error(
                            "HS triage parse failed for %s pass %s: %s (raw saved to %s)",
                            iso3_up,
                            pass_idx,
                            exc,
                            raw_path,
                        )

            hazards_dict = hazards_dict if isinstance(hazards_dict, dict) else {}
            hazard_codes = sorted({(hz_code or "").upper().strip() for hz_code in hazards_dict.keys()})
            logger.info(
                "HS triage hazards for %s pass %s: %d returned [%s]",
                iso3_up,
                pass_idx,
                len(hazard_codes),
                ", ".join(hazard_codes) if hazard_codes else "",
            )

            usage_for_log = dict(usage or {})
            if (usage_for_log.get("cost_usd") in (None, 0, 0.0)) and (usage_for_log.get("total_tokens") or 0):
                model_id_for_cost = getattr(model_spec, "model_id", None) or _resolve_hs_model()
                if model_id_for_cost:
                    usage_for_log["cost_usd"] = estimate_cost_usd(str(model_id_for_cost), usage_for_log)

            log_hs_llm_call(
                hs_run_id=run_id,
                iso3=iso3_up,
                hazard_code=f"pass_{pass_idx}",
                model_spec=model_spec,
                prompt_text=prompt,
                response_text=text or "",
                usage=usage_for_log,
                error_text=log_error_text,
            )
            logger.info(
                "HS logged triage call: hs_run_id=%s iso3=%s hazard=pass_%s",
                run_id,
                iso3_up,
                pass_idx,
            )

            pass_results.append(
                {
                    "text": text or "",
                    "usage": usage_for_log,
                    "triage": triage,
                    "hazards": hazards_dict,
                    "error_text": log_error_text,
                    "parse_ok": parse_ok,
                }
            )

        expected_hazards = list(hazard_catalog.keys())
        pass_one = _build_pass_hazards(pass_results[0]["hazards"], expected_hazards)
        pass_two = _build_pass_hazards(pass_results[1]["hazards"], expected_hazards)

        combined_hazards: Dict[str, Any] = {}
        for hz_code in expected_hazards:
            score1 = pass_one[hz_code]["score"]
            score2 = pass_two[hz_code]["score"]
            avg_score = (score1 + score2) / 2
            combined_hazards[hz_code] = {
                "triage_score": avg_score,
                "drivers": _merge_unique(pass_one[hz_code]["drivers"], pass_two[hz_code]["drivers"]),
                "regime_shifts": _merge_unique(
                    pass_one[hz_code]["regime_shifts"],
                    pass_two[hz_code]["regime_shifts"],
                ),
                "data_quality": {
                    "score_pass1": score1,
                    "score_pass2": score2,
                    "parse_ok1": bool(pass_results[0]["parse_ok"]),
                    "parse_ok2": bool(pass_results[1]["parse_ok"]),
                    "error1": _short_error(pass_results[0]["error_text"]),
                    "error2": _short_error(pass_results[1]["error_text"]),
                    "pass1": pass_one[hz_code]["data_quality"],
                    "pass2": pass_two[hz_code]["data_quality"],
                },
                "scenario_stub": pass_one[hz_code]["scenario_stub"] or pass_two[hz_code]["scenario_stub"],
            }

        triage = {"country": iso3_up, "hazards": combined_hazards}
        _write_hs_triage(run_id, iso3_up, triage)

        overall_error = None
        if not pass_results[0]["parse_ok"] and not pass_results[1]["parse_ok"]:
            overall_error = "triage parse failed for both passes"
            logger.error("HS triage error for %s: %s", iso3_up, overall_error)

        total_tokens = sum(result["usage"].get("total_tokens") or 0 for result in pass_results)
        total_cost = sum(result["usage"].get("cost_usd") or 0.0 for result in pass_results)
        logger.info(
            "HS triage completed for %s (%s): tokens=%s cost_usd=%.4f",
            country_name,
            iso3,
            total_tokens,
            total_cost,
        )

        response_text = "\n\n".join([result["text"] for result in pass_results if result["text"]])
        return {"iso3": iso3_up, "error_text": overall_error, "response_text": response_text}
    finally:
        con.close()


def _load_country_list(
    provided: list[str] | None,
) -> Tuple[list[Tuple[str, str]], list[dict[str, str]], list[str]]:
    iso3_to_name, name_to_iso3 = _load_country_registry()

    if provided:
        raw_countries = provided
    else:
        country_list_path = CURRENT_DIR / "hs_country_list.txt"
        with open(country_list_path, "r", encoding="utf-8") as f:
            raw_countries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    resolved: list[Tuple[str, str]] = []
    skipped_entries: list[dict[str, str]] = []
    for raw in raw_countries:
        name, iso3 = _resolve_country(raw, iso3_to_name, name_to_iso3)
        if iso3:
            resolved.append((name, iso3))
        else:
            skipped_entries.append(
                {
                    "raw": raw,
                    "normalized": _norm_country_key(raw),
                    "reason": "not_found",
                }
            )

    if skipped_entries:
        logger.warning(
            "HS country resolver: skipped %d unknown/invalid entries: %s",
            len(skipped_entries),
            [entry.get("raw", "") for entry in skipped_entries],
        )

    return resolved, skipped_entries, raw_countries


def main(countries: list[str] | None = None):
    logger.info("Starting Horizon Scanner triage run...")
    ensure_schema()

    start_time = datetime.utcnow()
    run_id = f"hs_{start_time.strftime('%Y%m%dT%H%M%S')}"
    os.environ["PYTHIA_HS_RUN_ID"] = run_id
    reset_provider_failures_for_run(run_id)

    country_entries, skipped_entries, requested_countries = _load_country_list(countries)
    if not country_entries:
        logger.warning("No countries supplied to Horizon Scanner; exiting.")
        if skipped_entries:
            logger.warning(
                "HS country resolver skipped %d unknown/invalid entries; see warnings above for details",
                len(skipped_entries),
            )
        print(f"HS_RUN_ID={run_id}")
        print("HS_RESOLVED_ISO3S=")
        return

    logger.info("Processing %d countries with max %d workers", len(country_entries), HS_MAX_WORKERS)

    triage_failures: list[_TriageCallResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=HS_MAX_WORKERS) as pool:
        futures = [
            pool.submit(_run_hs_for_country, run_id, iso3, name)
            for name, iso3 in country_entries
        ]

        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result and result.get("error_text"):
                    triage_failures.append(result)
            except Exception:
                logger.exception("HS triage failed for one country")

    iso3_list = [iso3 for (_name, iso3) in country_entries]
    try:
        git_sha = os.getenv("GITHUB_SHA") or ""
        config_profile = os.getenv("PYTHIA_CONFIG_PROFILE", "default")
        log_hs_run_to_db(
            run_id,
            iso3_list,
            git_sha=git_sha,
            config_profile=config_profile,
            requested_countries=requested_countries,
            skipped_entries=skipped_entries,
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("Failed to log hs_run %s: %s", run_id, exc)

    logger.info(
        "Horizon Scanner triage run complete for %d countries (skipped %d unknown/invalid entries)",
        len(country_entries),
        len(skipped_entries),
    )

    if triage_failures:
        diagnostics_dir = Path("diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        failures_path = diagnostics_dir / f"hs_triage_failures__{run_id}.json"
        try:
            failures_path.write_text(
                json.dumps(sorted(triage_failures, key=lambda r: r.get("iso3", "")), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Wrote HS triage failure diagnostics to %s", failures_path)
        except Exception:  # pragma: no cover - best-effort
            logger.exception("Failed to write HS triage failure diagnostics")

    resolved_iso3s = sorted({iso3 for iso3 in iso3_list})
    print(f"HS_RUN_ID={run_id}")
    print(f"HS_RESOLVED_ISO3S={','.join(resolved_iso3s)}")


if __name__ == "__main__":
    main()
