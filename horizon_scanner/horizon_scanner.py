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
    parse_ensemble_specs,
    reset_provider_failures_for_run,
)
from horizon_scanner.db_writer import (
    BLOCKED_HAZARDS,
    HAZARD_CONFIG,
    get_expected_hs_hazards,
    log_hs_country_reports_to_db,
    log_hs_run_to_db,
)
from horizon_scanner.regime_change import (
    coerce_regime_change,
    compute_level,
    compute_score,
    should_force_full_spd,
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
HS_FALLBACK_MODEL_SPECS = os.getenv("PYTHIA_HS_FALLBACK_MODEL_SPECS", "openai:gpt-5.1")
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

_HS_FALLBACK_SPECS: list[ModelSpec] = []


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
        f"{label} horizon scan web research focused on out-of-pattern/regime-change triggers and "
        "baseline continuation signals from the last 120 days across conflict, displacement, "
        "disasters, food security, and political stability. Recent signals must be bullets "
        "formatted like: TAIL-UP | month_2-3 | UP | <signal>; TAIL-DOWN | month_1-2 | DOWN | <signal>; "
        "BASELINE | month_1-6 | MIXED | <signal>. Provide observable trigger indicators "
        "(policy moves, troop mobilization, rainfall anomalies, river gauge alerts, border closures, "
        "camp saturation, UNHCR/IOM updates). Hazard trigger vocabulary: "
        "ACE: ceasefire collapse, major offensive, external intervention, election violence. "
        "DI: border policy, forced returns, new corridor, camp capacity, UNHCR/IOM updates. "
        "FL/TC: seasonal outlook, rainfall anomalies, SST/ENSO, landfall/track, dam releases. "
        "DR/HW: rainfall deficit, vegetation stress, heat index warnings, water rationing, food price spikes. "
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
        logger.debug("HS web research query for %s: %s", iso3, query)
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


def _resolve_hs_fallback_specs() -> list[ModelSpec]:
    specs = parse_ensemble_specs(HS_FALLBACK_MODEL_SPECS)
    return [spec for spec in specs if spec.active]


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


def _coerce_score_or_none(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


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


def _status_from_error(error_text: str | None) -> str:
    if not error_text:
        return "ok"
    lowered = str(error_text).lower()
    if "cooldown active" in lowered:
        return "cooldown"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "parse failed" in lowered or "json" in lowered:
        return "parse_error"
    if "empty response" in lowered:
        return "empty_response"
    return "provider_error"


def _merge_unique(values_a: list[str], values_b: list[str], limit: int = 6) -> list[str]:
    merged = sorted({value for value in values_a + values_b if value})
    return merged[:limit]


def _merge_unique_signals(
    signals_a: list[dict[str, Any]],
    signals_b: list[dict[str, Any]],
    limit: int = 6,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in signals_a + signals_b:
        if not isinstance(entry, dict):
            continue
        key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
        if len(merged) >= limit:
            break
    merged.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return merged[:limit]


async def _repair_hs_triage_json(
    raw_text: str,
    *,
    run_id: str | None = None,
    fallback_specs: list[ModelSpec],
) -> tuple[dict[str, Any], dict[str, Any], str, ModelSpec | None]:
    if not raw_text:
        return {}, {}, "empty response", None
    repair_prompt = (
        "Convert the following into valid JSON ONLY. No prose. Preserve keys/values. "
        "Output a single JSON object.\n\n"
        f"{raw_text}"
    )
    for spec in fallback_specs:
        text, usage, error = await call_chat_ms(
            spec,
            repair_prompt,
            temperature=0.0,
            prompt_key="hs.triage.json_repair",
            prompt_version="1.0.0",
            component="HorizonScanner",
            run_id=run_id,
        )
        if error:
            continue
        try:
            obj = _parse_hs_triage_json(text)
        except Exception as exc:  # noqa: BLE001
            return {}, usage or {}, f"triage repair parse failed: {type(exc).__name__}: {exc}", spec
        return obj, usage or {}, "", spec
    return {}, {}, "triage repair failed", None


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
            regime_change = coerce_regime_change(None)
            pass_hazards[hz_code] = {
                "score": None,
                "score_valid": False,
                "drivers": [],
                "regime_shifts": [],
                "data_quality": {"status": "missing_in_model_output"},
                "scenario_stub": "",
                "rc_valid": regime_change.get("valid", False),
                "rc_likelihood": regime_change.get("likelihood"),
                "rc_magnitude": regime_change.get("magnitude"),
                "rc_direction": regime_change.get("direction"),
                "rc_window": regime_change.get("window"),
                "rc_json": regime_change,
            }
            continue

        data_quality = hdata.get("data_quality") or {}
        if not isinstance(data_quality, dict):
            data_quality = {"raw": data_quality}

        score_value = _coerce_score_or_none(hdata.get("triage_score"))
        regime_change = coerce_regime_change(hdata.get("regime_change"))
        pass_hazards[hz_code] = {
            "score": score_value,
            "score_valid": score_value is not None,
            "drivers": _coerce_list(hdata.get("drivers")),
            "regime_shifts": _coerce_list(hdata.get("regime_shifts")),
            "data_quality": data_quality,
            "scenario_stub": str(hdata.get("scenario_stub") or ""),
            "rc_valid": regime_change.get("valid", False),
            "rc_likelihood": regime_change.get("likelihood"),
            "rc_magnitude": regime_change.get("magnitude"),
            "rc_direction": regime_change.get("direction"),
            "rc_window": regime_change.get("window"),
            "rc_json": regime_change,
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


def _is_missing_regime_change_column_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "regime_change" in message and (
        "column" in message or "no such column" in message or "does not exist" in message
    )


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
                regime_change = coerce_regime_change(None)
            else:
                score = float(hdata.get("triage_score") or 0.0)
                drivers = hdata.get("drivers") or []
                regime_shifts = hdata.get("regime_shifts") or []
                data_quality = hdata.get("data_quality") or {}
                if not isinstance(data_quality, dict):
                    data_quality = {"raw": data_quality}
                scenario_stub = hdata.get("scenario_stub") or ""
                regime_change = coerce_regime_change(hdata.get("regime_change"))

            tier = tier_from_score(score)
            regime_change_score = compute_score(
                regime_change.get("likelihood"), regime_change.get("magnitude")
            )
            regime_change_level = compute_level(
                regime_change.get("likelihood"),
                regime_change.get("magnitude"),
                regime_change_score,
            )
            force_full_spd = should_force_full_spd(regime_change_level, regime_change_score)
            need_full_spd = tier in {"priority", "watchlist"} or force_full_spd
            logger.debug(
                "HS triage regime change | run_id=%s iso3=%s hazard=%s likelihood=%s magnitude=%s "
                "direction=%s window=%s score=%.3f level=%s force_full_spd=%s",
                run_id,
                iso3_up,
                hz_up,
                regime_change.get("likelihood"),
                regime_change.get("magnitude"),
                regime_change.get("direction"),
                regime_change.get("window"),
                regime_change_score,
                regime_change_level,
                force_full_spd,
            )

            insert_payload = [
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
                regime_change.get("likelihood"),
                regime_change.get("magnitude"),
                regime_change_score,
                regime_change_level,
                regime_change.get("direction"),
                regime_change.get("window"),
                json.dumps(regime_change),
            ]
            try:
                con.execute(
                    """
                    INSERT INTO hs_triage (
                        run_id, iso3, hazard_code,
                        tier, triage_score, need_full_spd,
                        drivers_json, regime_shifts_json, data_quality_json, scenario_stub,
                        regime_change_likelihood, regime_change_magnitude, regime_change_score,
                        regime_change_level, regime_change_direction, regime_change_window,
                        regime_change_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_payload,
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_missing_regime_change_column_error(exc):
                    raise
                logger.warning(
                    "HS triage: regime_change columns missing; retrying legacy insert | run_id=%s iso3=%s hazard=%s",
                    run_id,
                    iso3_up,
                    hz_up,
                )
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
    response_text: str
    pass_results: list[dict[str, Any]]
    final_status: str
    pass1_status: str
    pass2_status: str
    pass1_valid: bool
    pass2_valid: bool
    primary_model_id: str
    fallback_model_id: str | None
    error_text: str | None


async def _call_hs_model(
    prompt_text: str,
    *,
    run_id: str | None = None,
    fallback_specs: list[ModelSpec] | None = None,
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
            prompt_version="1.1.0",
            component="HorizonScanner",
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", spec

    usage = usage or {}
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    if not error:
        usage["fallback_used"] = False
        usage["model_selected"] = f"{spec.provider}:{spec.model_id}"
        return text, usage, error, spec

    primary_error = _short_error(error)
    fallback_specs = fallback_specs or []
    for fallback_spec in fallback_specs:
        fallback_start = time.time()
        try:
            fallback_text, fallback_usage, fallback_error = await call_chat_ms(
                fallback_spec,
                prompt_text,
                temperature=HS_TEMPERATURE,
                prompt_key="hs.triage.v2",
                prompt_version="1.1.0",
                component="HorizonScanner",
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_error = f"provider call error: {exc}"
            fallback_text = ""
            fallback_usage = {}
        fallback_usage = fallback_usage or {}
        fallback_usage.setdefault("elapsed_ms", int((time.time() - fallback_start) * 1000))
        fallback_usage["fallback_used"] = True
        fallback_usage["primary_error"] = primary_error
        fallback_usage["model_selected"] = f"{fallback_spec.provider}:{fallback_spec.model_id}"
        if not fallback_error:
            return fallback_text, fallback_usage, "", fallback_spec
    usage["fallback_used"] = True
    usage["primary_error"] = primary_error
    usage["model_selected"] = f"{spec.provider}:{spec.model_id}"
    return "", usage, error, spec


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
        fallback_specs = _HS_FALLBACK_SPECS or _resolve_hs_fallback_specs()
        for pass_idx in (1, 2):
            call_start = time.time()
            text, usage, error, model_spec = asyncio.run(
                _call_hs_model(prompt, run_id=run_id, fallback_specs=fallback_specs)
            )
            usage = usage or {}
            usage.setdefault("elapsed_ms", int((time.time() - call_start) * 1000))

            log_error_text: str | None = None
            triage: Dict[str, Any] = {}
            hazards_dict: Dict[str, Any] = {}
            parse_ok = True
            json_repair_used = False
            repair_error: str | None = None
            repair_model_selected: str | None = None

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
                        repair_obj, repair_usage, repair_error, repair_spec = asyncio.run(
                            _repair_hs_triage_json(raw, run_id=run_id, fallback_specs=fallback_specs)
                        )
                        if not repair_error:
                            triage = repair_obj
                            hazards_dict = triage.get("hazards") or {}
                            parse_ok = True
                            json_repair_used = True
                            log_error_text = None
                            if repair_spec:
                                repair_model_selected = f"{repair_spec.provider}:{repair_spec.model_id}"
                            if repair_usage:
                                usage.setdefault("repair_usage", repair_usage)
                        else:
                            repair_model_selected = (
                                f"{repair_spec.provider}:{repair_spec.model_id}" if repair_spec else None
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
            if json_repair_used:
                usage_for_log["json_repair_used"] = True
                if repair_model_selected:
                    usage_for_log["repair_model_selected"] = repair_model_selected
            if repair_error:
                usage_for_log["repair_error"] = _short_error(repair_error)
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
                    "fallback_used": bool(usage_for_log.get("fallback_used")),
                    "primary_error": usage_for_log.get("primary_error"),
                    "json_repair_used": json_repair_used,
                    "repair_error": repair_error,
                    "model_selected": usage_for_log.get("model_selected"),
                }
            )

        expected_hazards = list(hazard_catalog.keys())
        pass_one = _build_pass_hazards(pass_results[0]["hazards"], expected_hazards)
        pass_two = _build_pass_hazards(pass_results[1]["hazards"], expected_hazards)

        combined_hazards: Dict[str, Any] = {}
        for hz_code in expected_hazards:
            score1 = pass_one[hz_code]["score"]
            score2 = pass_two[hz_code]["score"]
            score1_valid = bool(pass_one[hz_code].get("score_valid"))
            score2_valid = bool(pass_two[hz_code].get("score_valid"))
            if score1_valid and score2_valid:
                avg_score = (float(score1) + float(score2)) / 2
                combined_status = "ok"
            elif score1_valid:
                avg_score = float(score1)
                combined_status = "degraded"
            elif score2_valid:
                avg_score = float(score2)
                combined_status = "degraded"
            else:
                avg_score = 0.0
                combined_status = "error"
            rc1_valid = bool(pass_one[hz_code].get("rc_valid"))
            rc2_valid = bool(pass_two[hz_code].get("rc_valid"))
            rc1_likelihood = pass_one[hz_code].get("rc_likelihood")
            rc2_likelihood = pass_two[hz_code].get("rc_likelihood")
            rc1_magnitude = pass_one[hz_code].get("rc_magnitude")
            rc2_magnitude = pass_two[hz_code].get("rc_magnitude")
            rc1_direction = pass_one[hz_code].get("rc_direction") or "unclear"
            rc2_direction = pass_two[hz_code].get("rc_direction") or "unclear"
            rc1_window = pass_one[hz_code].get("rc_window") or ""
            rc2_window = pass_two[hz_code].get("rc_window") or ""

            if rc1_valid and rc2_valid:
                rc_likelihood = (float(rc1_likelihood) + float(rc2_likelihood)) / 2
                rc_magnitude = (float(rc1_magnitude) + float(rc2_magnitude)) / 2
                rc_valid = True
                rc_status = "ok"
                if rc1_direction == "unclear" or rc2_direction == "unclear":
                    rc_direction = "unclear"
                elif rc1_direction == rc2_direction:
                    rc_direction = rc1_direction
                else:
                    rc_direction = "mixed"
                if rc1_window == rc2_window:
                    rc_window = rc1_window
                elif rc1_window and not rc2_window:
                    rc_window = rc1_window
                elif rc2_window and not rc1_window:
                    rc_window = rc2_window
                else:
                    rc_window = ""
            elif rc1_valid:
                rc_likelihood = float(rc1_likelihood)
                rc_magnitude = float(rc1_magnitude)
                rc_valid = True
                rc_status = "degraded"
                rc_direction = rc1_direction
                rc_window = rc1_window
            elif rc2_valid:
                rc_likelihood = float(rc2_likelihood)
                rc_magnitude = float(rc2_magnitude)
                rc_valid = True
                rc_status = "degraded"
                rc_direction = rc2_direction
                rc_window = rc2_window
            else:
                rc_likelihood = 0.0
                rc_magnitude = 0.0
                rc_valid = False
                rc_status = "error"
                rc_direction = "unclear"
                rc_window = ""

            rc1_json = pass_one[hz_code].get("rc_json") or {}
            rc2_json = pass_two[hz_code].get("rc_json") or {}
            rc_bullets = _merge_unique(
                rc1_json.get("rationale_bullets") or [],
                rc2_json.get("rationale_bullets") or [],
            )
            rc_signals = _merge_unique_signals(
                rc1_json.get("trigger_signals") or [],
                rc2_json.get("trigger_signals") or [],
            )

            combined_hazards[hz_code] = {
                "triage_score": avg_score,
                "drivers": _merge_unique(pass_one[hz_code]["drivers"], pass_two[hz_code]["drivers"]),
                "regime_shifts": _merge_unique(
                    pass_one[hz_code]["regime_shifts"],
                    pass_two[hz_code]["regime_shifts"],
                ),
                "regime_change": {
                    "likelihood": rc_likelihood,
                    "direction": rc_direction,
                    "magnitude": rc_magnitude,
                    "window": rc_window,
                    "rationale_bullets": rc_bullets,
                    "trigger_signals": rc_signals,
                    "valid": rc_valid,
                    "status": rc_status,
                },
                "data_quality": {
                    "status": combined_status,
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
        pass1_status = _status_from_error(pass_results[0]["error_text"])
        pass2_status = _status_from_error(pass_results[1]["error_text"])
        if pass1_status == "ok" and pass_results[0].get("fallback_used"):
            pass1_status = "fallback_used"
        if pass2_status == "ok" and pass_results[1].get("fallback_used"):
            pass2_status = "fallback_used"
        if pass1_status == "ok" and pass_results[0].get("json_repair_used"):
            pass1_status = "fallback_used"
        if pass2_status == "ok" and pass_results[1].get("json_repair_used"):
            pass2_status = "fallback_used"
        pass1_valid = any(pass_one[hz].get("score_valid") for hz in expected_hazards)
        pass2_valid = any(pass_two[hz].get("score_valid") for hz in expected_hazards)
        if pass1_valid and pass2_valid:
            final_status = "ok"
        elif pass1_valid or pass2_valid:
            final_status = "degraded"
        else:
            final_status = "failed"
        fallback_model_id = None
        if pass_results[0].get("fallback_used") or pass_results[1].get("fallback_used"):
            if fallback_specs:
                fallback_model_id = fallback_specs[0].model_id
        return {
            "iso3": iso3_up,
            "error_text": overall_error,
            "response_text": response_text,
            "pass_results": pass_results,
            "final_status": final_status,
            "pass1_status": pass1_status,
            "pass2_status": pass2_status,
            "pass1_valid": pass1_valid,
            "pass2_valid": pass2_valid,
            "primary_model_id": _resolve_hs_model(),
            "fallback_model_id": fallback_model_id,
        }
    finally:
        con.close()


def _load_country_list(
    provided: list[str] | None,
) -> Tuple[list[Tuple[str, str]], list[dict[str, str]], list[str]]:
    iso3_to_name, name_to_iso3 = _load_country_registry()

    if provided:
        raw_countries = provided
    else:
        env_only = os.getenv("PYTHIA_HS_ONLY_COUNTRIES", "").strip()
        if env_only:
            raw_countries = [part.strip() for part in env_only.split(",") if part.strip()]
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
    fallback_specs = _resolve_hs_fallback_specs()
    if not fallback_specs:
        raise SystemExit(
            "HS fallback model required but not active; check OPENAI_API_KEY / config"
        )
    global _HS_FALLBACK_SPECS
    _HS_FALLBACK_SPECS = fallback_specs

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

    triage_results: list[_TriageCallResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=HS_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_hs_for_country, run_id, iso3, name): (iso3, name)
            for name, iso3 in country_entries
        }

        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result:
                    triage_results.append(result)
            except Exception:
                iso3, _name = futures.get(fut, ("", ""))
                logger.exception("HS triage failed for one country")
                if iso3:
                    triage_results.append(
                        {
                            "iso3": iso3.upper(),
                            "error_text": "triage execution failed",
                            "response_text": "",
                            "pass_results": [],
                            "final_status": "failed",
                            "pass1_status": "provider_error",
                            "pass2_status": "provider_error",
                            "pass1_valid": False,
                            "pass2_valid": False,
                            "primary_model_id": _resolve_hs_model(),
                            "fallback_model_id": fallback_specs[0].model_id if fallback_specs else None,
                        }
                    )

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

    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    failures_payload: list[dict[str, Any]] = []
    for result in sorted(triage_results, key=lambda r: r.get("iso3", "")):
        coverage_rows.append(
            {
                "iso3": result.get("iso3"),
                "pass1_status": result.get("pass1_status"),
                "pass2_status": result.get("pass2_status"),
                "primary_model_id": result.get("primary_model_id"),
                "fallback_model_id": result.get("fallback_model_id"),
                "final_status": result.get("final_status"),
            }
        )
        if result.get("final_status") != "ok":
            failures_payload.append(
                {
                    "iso3": result.get("iso3"),
                    "final_status": result.get("final_status"),
                    "pass1_status": result.get("pass1_status"),
                    "pass2_status": result.get("pass2_status"),
                    "pass1_valid": result.get("pass1_valid"),
                    "pass2_valid": result.get("pass2_valid"),
                    "primary_model_id": result.get("primary_model_id"),
                    "fallback_model_id": result.get("fallback_model_id"),
                    "pass_results": result.get("pass_results", []),
                }
            )

    coverage_path = diagnostics_dir / f"hs_triage_coverage__{run_id}.csv"
    try:
        with open(coverage_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iso3",
                    "pass1_status",
                    "pass2_status",
                    "primary_model_id",
                    "fallback_model_id",
                    "final_status",
                ],
            )
            writer.writeheader()
            writer.writerows(coverage_rows)
        logger.info("Wrote HS triage coverage report to %s", coverage_path)
    except Exception:  # pragma: no cover - best-effort
        logger.exception("Failed to write HS triage coverage report")

    failures_path = diagnostics_dir / f"hs_triage_failures__{run_id}.json"
    try:
        failures_path.write_text(
            json.dumps(failures_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote HS triage failure diagnostics to %s", failures_path)
    except Exception:  # pragma: no cover - best-effort
        logger.exception("Failed to write HS triage failure diagnostics")

    resolved_iso3s = sorted({iso3 for iso3 in iso3_list})
    print(f"HS_RUN_ID={run_id}")
    print(f"HS_RESOLVED_ISO3S={','.join(resolved_iso3s)}")
    ok_iso3s = sorted(
        [result.get("iso3") for result in triage_results if result.get("final_status") == "ok"]
    )
    degraded_iso3s = sorted(
        [result.get("iso3") for result in triage_results if result.get("final_status") == "degraded"]
    )
    failed_iso3s = sorted(
        [result.get("iso3") for result in triage_results if result.get("final_status") == "failed"]
    )
    rerun_iso3s = sorted({*degraded_iso3s, *failed_iso3s})
    print(f"HS_TRIAGE_OK_ISO3S={','.join([iso3 for iso3 in ok_iso3s if iso3])}")
    print(f"HS_TRIAGE_DEGRADED_ISO3S={','.join([iso3 for iso3 in degraded_iso3s if iso3])}")
    print(f"HS_TRIAGE_FAILED_ISO3S={','.join([iso3 for iso3 in failed_iso3s if iso3])}")
    print(f"HS_TRIAGE_RERUN_ISO3S={','.join([iso3 for iso3 in rerun_iso3s if iso3])}")
    print(f"HS_TRIAGE_COVERAGE_CSV={coverage_path}")


if __name__ == "__main__":
    main()
