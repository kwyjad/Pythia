# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

# ANCHOR: prompts (paste whole file)
from __future__ import annotations
import importlib
import importlib.util
import logging
import os
import threading
import re
from datetime import date, datetime
import functools
from typing import Any, Dict, Optional
import json
from pythia.buckets import labels_for, n_buckets_for

from .config import CALIBRATION_PATH, ist_date
from .hazard_prompts import get_hazard_reasoning_block

LOG = logging.getLogger(__name__)



# NMME anomalies are stored in their own units (resolver.ingestion.nmme.UNITS).
# Until Oct 2026 this note said "sigma", which they never were.
NMME_UNITS_NOTE = (
    "Note: Anomalies are departures from the model climatology: temperature in °C, "
    "precipitation in mm/day (0.5 mm/day is about 15 mm a month). They are not "
    "standardised, so a dry country's precipitation anomaly is small in absolute terms."
)

def _json_dumps_for_prompt(obj: Any, **kwargs: Any) -> str:
    """
    JSON-encode helper for prompts that tolerates Python objects like date
    by stringifying unknown types via default=str.
    """

    return json.dumps(obj, default=str, **kwargs)


def _self_search_enabled() -> bool:
    """Mirror of the executor's self-search gate.

    Prompt text offering the NEED_WEB_EVIDENCE escape must agree with
    forecaster.self_search.self_search_enabled(), otherwise models are
    invited to request evidence the pipeline refuses to fetch and their
    forecast is dropped with error `self_search_disabled`.
    """
    try:
        from forecaster.self_search import self_search_enabled
        return bool(self_search_enabled())
    except Exception:
        return False

_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)


def _pythia_db_url_from_config() -> Optional[str]:
    try:
        if _PYTHIA_CFG_LOAD is None:
            return None
        cfg = _PYTHIA_CFG_LOAD()
        app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
        db_url = str(app_cfg.get("db_url", "")).strip()
        return db_url or None
    except Exception:
        return None

def _load_calibration_note() -> str:
    """
    Pull the latest calibration guidance from DuckDB (calibration_advice).
    Returns "" if nothing readable is found, so prompts stay valid.
    """
    txt = ""
    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            row = con.execute(
                """
                SELECT advice
                FROM calibration_advice
                ORDER BY as_of_month DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            duckdb_io.close_db(con)
        if row and row[0]:
            txt = str(row[0])
    except Exception:
        txt = ""

    if not txt and CALIBRATION_PATH:
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                txt = f.read().strip()
        except Exception:
            txt = ""

    if not txt:
        return ""
    return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")

_MEMBER_ADVICE_CACHE: dict[tuple, str] = {}
_MEMBER_ADVICE_LOCK = threading.Lock()


def reset_member_calibration_advice_cache() -> None:
    """Clear cached per-member advice (called with the weights-cache reset)."""
    with _MEMBER_ADVICE_LOCK:
        _MEMBER_ADVICE_CACHE.clear()


# --- Advice carry-over, the no-advice arm (PR feat/family-advice-and-recalibration)

#: Stored names that are not model ids but stand for one: Track 2's single
#: forecast is stored as ``track2_flash`` and is made by the flash model of
#: the ``track2_spd`` role, so it takes that family's advice.
ADVICE_FAMILY_ALIASES = {"track2_flash": "gemini_flash"}
#: Distinct questions of a model's own before its exact advice is used and
#: carried family advice stops.
ADVICE_OWN_QUESTIONS = 20
ADVICE_TENDENCY_LINE = "Treat these as tendencies to check, not corrections to apply mechanically."


def advice_family_carryover_enabled() -> bool:
    """``PYTHIA_ADVICE_FAMILY_CARRYOVER`` (default off)."""
    return os.getenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "0").strip().lower() in ("1", "true", "yes")


def advice_experiment_share() -> float:
    """``PYTHIA_ADVICE_EXPERIMENT_SHARE`` clamped to [0, 1] (default 0)."""
    try:
        v = float(os.getenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0") or 0)
    except ValueError:
        return 0.0
    return min(max(v, 0.0), 1.0)


def advice_arm(question_id: Optional[str]) -> Optional[str]:
    """``"no_advice"`` or ``"advice"`` for a question, None when the experiment is off.

    The arm is a pure function of the question id (``sha1(qid)[:8]`` as a
    fraction of 2^32 against the share), so a rerun lands in the same arm and
    nothing has to be stored before the prompt is built.
    """
    share = advice_experiment_share()
    if share <= 0 or not question_id:
        return None
    import hashlib

    frac = int(hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "no_advice" if frac < share else "advice"


def advice_family_for(model_name: Optional[str]) -> Optional[str]:
    name = (model_name or "").strip()
    if name in ADVICE_FAMILY_ALIASES:
        return ADVICE_FAMILY_ALIASES[name]
    try:
        from pythia.llm_profiles import model_family

        return model_family(name)
    except Exception:  # noqa: BLE001
        return None


def _advice_rows(hz: str, m: str, names: list[str]) -> dict[str, dict]:
    """{model_name: findings} for the newest advice row of each name."""
    out: dict[str, dict] = {}
    advice_version = os.getenv("PYTHIA_ADVICE_VERSION", "").strip() or None
    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            version_clause = " AND advice_version = ?" if advice_version else ""
            for name in names:
                params = [hz, m, name] + ([advice_version] if advice_version else [])
                row = con.execute(
                    f"""
                    SELECT findings_json FROM calibration_advice
                    WHERE hazard_code = ? AND metric = ? AND model_name = ?{version_clause}
                    ORDER BY as_of_month DESC LIMIT 1
                    """,
                    params,
                ).fetchone()
                if row and row[0]:
                    try:
                        out[name] = json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
                    except Exception:  # noqa: BLE001
                        continue
        finally:
            duckdb_io.close_db(con)
    except Exception:  # noqa: BLE001
        return out
    return out


def render_advice_observations(
    findings: dict,
    *,
    header: str,
    weight: float = 1.0,
    drop_buckets: bool = False,
    drop_prior: bool = False,
) -> str:
    """Advice as OBSERVATIONS ("you assigned 9%, observed 34%"), never orders.

    ``weight`` below 1 shrinks each carried gap toward what was assigned and
    says so. ``drop_buckets`` leaves out per-bucket numbers where family
    recalibration already corrects them; ``drop_prior`` leaves out prior
    anchoring where the prompt now hands the member its prior.
    """
    lines: list[str] = [header]

    def _obs(a: float, o: float) -> float:
        return a + (o - a) * weight

    tc = findings.get("tail_coverage") or {}
    if tc and not drop_buckets:
        a = 100 * float(tc.get("avg_assigned_tail") or 0)
        o = 100 * _obs(float(tc.get("avg_assigned_tail") or 0), float(tc.get("actual_tail_rate") or 0))
        lines.append(f"- Top two buckets: you assigned {a:.0f}% on average; observed {o:.0f}%.")
    bc = findings.get("bucket_calibration") or []
    if bc and not drop_buckets:
        for e in bc:
            a = float(e.get("mean_assigned") or 0)
            o = _obs(a, float(e.get("actual_rate") or 0))
            if abs(a - o) * 100 < 3:
                continue
            lines.append(
                f"- Bucket {e.get('class_bin')}: you assigned {100 * a:.0f}%; observed {100 * o:.0f}%."
            )
    hd = findings.get("horizon_diff") or {}
    if hd and hd.get("flat"):
        lines.append(
            "- Your month-1 and month-6 distributions were nearly identical "
            f"(JS divergence {float(hd.get('jsd_m1_m6') or 0):.4f})."
        )
    pa = findings.get("prior_anchoring") or {}
    worst = pa.get("worst_bucket_gap") or {}
    if worst and not drop_prior and abs(float(worst.get("gap_pp") or 0)) > 5:
        gap = float(worst.get("gap_pp") or 0) * weight
        more = "more" if gap > 0 else "less"
        lines.append(
            f"- Your declared priors put {abs(gap):.0f} points {more} on bucket "
            f"{worst.get('bucket')} than resolved outcomes did."
        )
    if len(lines) == 1:
        return ""
    if weight < 1.0:
        lines.append(
            f"- Observed figures above are shown at {weight:.0%} of the measured gap, "
            "because you now have scored questions of your own."
        )
    lines.append(ADVICE_TENDENCY_LINE)
    return "\n".join(lines)


def _structured_member_advice(hz: str, m: str, name: str) -> str:
    """The member note under family carry-over or applied recalibration."""
    family = advice_family_for(name)
    names = [name] + ([f"family:{family}"] if family else [])
    rows = _advice_rows(hz, m, names)
    drop_prior = prior_anchor_enabled() and hz == "ACE" and m == "FATALITIES"
    drop_buckets = False
    try:
        from pythia.tools import family_recalibration as fr

        if fr.recalibration_mode() == "apply":
            brbv = None
            if drop_prior:
                brbv = prior_anchor_version()
            info = fr.lookup(name, hz, m, base_rate_block_version=brbv,
                             rc_guidance=rc_guidance_version(track=1))
            drop_buckets = info.get("mode") == "apply"
    except Exception:  # noqa: BLE001
        drop_buckets = False
    exact = rows.get(name) or {}
    if int(exact.get("n_questions") or 0) >= ADVICE_OWN_QUESTIONS:
        return render_advice_observations(
            exact,
            header=f"From your own {int(exact['n_questions'])} scored questions on this hazard and metric:",
            drop_buckets=drop_buckets,
            drop_prior=drop_prior,
        )
    if not advice_family_carryover_enabled() or not family:
        return ""
    fam = rows.get(f"family:{family}") or {}
    if not fam:
        return ""
    ids = fam.get("contributing_ids") or {}
    own = int(ids.get(name) or 0)
    others = {k: v for k, v in ids.items() if k != name}
    if not others:
        return ""
    weight = 1.0 - own / ADVICE_OWN_QUESTIONS if 0 < own < ADVICE_OWN_QUESTIONS else 1.0
    who = ", ".join(sorted(others))
    header = (
        f"Carried from earlier versions of your model line ({who}; "
        f"{int(fam.get('n_questions') or 0)} scored questions on this hazard and metric):"
    )
    return render_advice_observations(
        fam, header=header, weight=weight, drop_buckets=drop_buckets, drop_prior=drop_prior,
    )


def load_member_calibration_advice(
    hazard_code: str, metric: str, model_name: str, question_id: Optional[str] = None,
) -> str:
    """The per-model advice for one ensemble member, or "".

    The shared (hazard, metric) advice lives in the prompt's cached prefix
    and is the same for every member. This is the member's OWN part, keyed by
    the model name its scores are stored under, and is appended to the tail
    of that member's prompt only. It was generated every cycle and read by
    nothing until Oct 2026: Track 1 builds one prompt for all members, and the
    only caller that passed a model name was Track 2, whose name is an
    aggregate that never has advice.
    """
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    name = (model_name or "").strip()
    if not hz or not m or not name or os.getenv("PYTHIA_MEMBER_ADVICE", "1") == "0":
        return ""
    if _advice_blocked(hz, m):
        return ""
    if advice_arm(question_id) == "no_advice":
        return ""
    structured = advice_family_carryover_enabled()
    if not structured:
        try:
            from pythia.tools import family_recalibration as _fr

            structured = _fr.recalibration_mode() == "apply"
        except Exception:  # noqa: BLE001
            structured = False
    key = (hz, m, name, "structured" if structured else "text")
    with _MEMBER_ADVICE_LOCK:
        if key in _MEMBER_ADVICE_CACHE:
            return _MEMBER_ADVICE_CACHE[key]

    if structured:
        try:
            text = _structured_member_advice(hz, m, name)
        except Exception:  # noqa: BLE001
            text = ""
        with _MEMBER_ADVICE_LOCK:
            _MEMBER_ADVICE_CACHE[key] = text
        return text

    text = ""
    advice_version = os.getenv("PYTHIA_ADVICE_VERSION", "").strip() or None
    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            version_clause = " AND advice_version = ?" if advice_version else ""
            params = [hz, m, name] + ([advice_version] if advice_version else [])
            row = con.execute(
                f"""
                SELECT advice
                FROM calibration_advice
                WHERE hazard_code = ? AND metric = ? AND model_name = ?
                  {version_clause}
                ORDER BY as_of_month DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row and row[0]:
                text = str(row[0])
                if len(text) > 2000:
                    text = text[:1900] + "\n…[truncated]"
        finally:
            duckdb_io.close_db(con)
    except Exception:
        text = ""

    with _MEMBER_ADVICE_LOCK:
        _MEMBER_ADVICE_CACHE[key] = text
    return text


def render_member_calibration_advice(advice: str) -> str:
    """The block appended to one member's prompt; "" when there is none."""
    if not advice:
        return ""
    return (
        "\n\nCALIBRATION NOTE FOR THIS MODEL (auto-generated from your own "
        "scored forecasts on this hazard and metric; it applies to you, not "
        "to the other forecasters):\n"
        + advice
        + "\n--- end model calibration note ---\n"
        "Apply this note while following the method above, then produce ONLY the "
        "JSON object specified in the Output instructions.\n"
    )


def _advice_blocked(hazard_code: str, metric: str) -> bool:
    """True when ``PYTHIA_ADVICE_BLOCK_GROUPS`` lists ``HAZARD/METRIC``.

    Same format as ``pythia.tools.generate_calibration_advice
    .advice_blocked_groups`` (comma-separated ``HAZARD/METRIC``); parsed here
    so the prompt builder does not import the advice generator. Used to
    withhold advice learned from outcomes known to be wrong — ACE/FATALITIES
    in Sept 2026, which had resolved to battles-only counts.
    """
    raw = os.getenv("PYTHIA_ADVICE_BLOCK_GROUPS", "") or ""
    key = f"{(hazard_code or '').strip().upper()}/{(metric or '').strip().upper()}"
    return any(
        part.strip().upper() == key for part in raw.split(",") if part.strip()
    )


def _load_calibration_advice_for_hazard(
    hazard_code: str,
    metric: str,
    model_name: Optional[str] = None,
) -> str:
    """Load calibration advice, preferring model-specific if available.

    Fallback chain:
      1. Shared advice for (hazard_code, metric, '__shared__')
         + Per-model advice for (hazard_code, metric, model_name)
      2. Global advice for ('*', '*', '__shared__')
      3. Empty string

    A group listed in ``PYTHIA_ADVICE_BLOCK_GROUPS`` gets nothing at all.
    Until Sept 2026 a third step returned "any most-recent row regardless of
    hazard", so a flood question with no advice of its own could be shown
    conflict advice; that step is gone.
    """
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    if _advice_blocked(hz, m):
        return ""

    # Read experiment version from env (default: any version)
    advice_version = os.getenv("PYTHIA_ADVICE_VERSION", "").strip() or None

    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            version_clause = ""
            version_params: list = []
            if advice_version:
                version_clause = " AND advice_version = ?"
                version_params = [advice_version]

            parts: list[str] = []

            # Always load shared advice first
            shared_row = con.execute(
                f"""
                SELECT advice
                FROM calibration_advice
                WHERE hazard_code = ? AND metric = ? AND model_name = '__shared__'
                  {version_clause}
                ORDER BY as_of_month DESC
                LIMIT 1
                """,
                [hz, m] + version_params,
            ).fetchone()

            if shared_row and shared_row[0]:
                parts.append(str(shared_row[0]))

            # Then load model-specific advice if available
            if model_name:
                model_row = con.execute(
                    f"""
                    SELECT advice
                    FROM calibration_advice
                    WHERE hazard_code = ? AND metric = ? AND model_name = ?
                      {version_clause}
                    ORDER BY as_of_month DESC
                    LIMIT 1
                    """,
                    [hz, m, model_name] + version_params,
                ).fetchone()

                if model_row and model_row[0]:
                    parts.append(str(model_row[0]))

            if parts:
                txt = "\n\n".join(parts)
                return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")

            # Fallback to global
            global_row = con.execute(
                f"""
                SELECT advice
                FROM calibration_advice
                WHERE hazard_code = '*' AND metric = '*' AND model_name = '__shared__'
                  {version_clause}
                ORDER BY as_of_month DESC
                LIMIT 1
                """,
                version_params,
            ).fetchone()

            if global_row and global_row[0]:
                txt = str(global_row[0])
                return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")
        finally:
            duckdb_io.close_db(con)
    except Exception:
        pass
    return ""


_CAL_NOTE = _load_calibration_note()
_CAL_PREFIX = (
    "CALIBRATION GUIDANCE (auto-generated weekly):\n"
    + (_CAL_NOTE if _CAL_NOTE else "(none available yet)")
    + "\n— end calibration —\n\n"
)


RESEARCH_V2_REQUIRED_OUTPUT_SCHEMA = """
```json
{
  "base_rate": {
    "qualitative_summary": "...",
    "resolver_support": {
      "recent_level": "low|medium|high",
      "trend": "up|down|flat|uncertain",
      "data_quality": "low|medium|high",
      "notes": "..."
    },
    "external_support": {
      "consensus": "increasing|decreasing|mixed|uncertain",
      "data_quality": "low|medium|high",
      "recent_analyses": ["..."]
    }
  },
  "update_signals": [
    {"description": "...", "direction": "up|down|unclear", "confidence": 0.7, "timeframe_months": 6, "sources": ["..."]}
  ],
  "regime_shift_signals": [
    {"description": "...", "likelihood": "low|medium|high", "timeframe_months": 3, "sources": ["..."]}
  ],
  "data_gaps": ["..."],
  "sources": ["url1", "url2"],
  "grounded": true
}
```
""".strip()


def build_scoring_resolution_block(
    *,
    hazard_code: str,
    metric: str,
    resolution_source: Optional[str] = None,
) -> str:
    """
    Build a short, LLM-friendly 'SCORING & RESOLUTION' block.

    Uses hazard_code + metric + optional resolution_source to explain:
      - What "affected" means for this question.
      - Which source is used (IFRC Montandon, IDMC/DTM, ACLED).
      - That we use Brier scores on the SPD buckets.

    This is inserted early in the SPD prompt.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    src = (resolution_source or "").upper()

    if m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        meaning = (
            "“affected” means people who are internally displaced (IDPs), "
            "as recorded by the Internal Displacement Monitoring Centre (IDMC), "
            "with IOM Displacement Tracking Matrix (DTM) as a fallback source."
        )
        source_label = "IDMC/DTM displacement"
    elif m == "FATALITIES" or "ACLED" in src:
        meaning = (
            "“affected” means conflict fatalities recorded by ACLED, summed over all "
            "ACLED event types (armed conflict event data)."
        )
        source_label = "ACLED conflict fatalities"
    else:
        meaning = (
            "“affected” means people affected by the hazard as recorded by "
            "IFRC Montandon (the Global Crisis Data Bank) for this hazard and country."
        )
        source_label = "IFRC Montandon people affected"

    lines = []
    lines.append("SCORING & RESOLUTION")
    lines.append("")
    lines.append(
        "- All forecasts will be resolved using Resolver's canonical metric for this "
        "hazard and country, and scored using **Brier scores** on your SPD buckets."
    )
    lines.append("- For this question:")
    lines.append(f"  - {meaning}")
    lines.append(f"  - Source for resolution: {source_label}.")
    lines.append("")
    return "\n".join(lines)


def build_time_horizon_block(
    *,
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]] = None,
    hazard_code: str,
    metric: str,
    resolution_source: Optional[str] = None,
) -> str:
    """
    Build a 'TIME HORIZON & RESOLUTION' block explaining:
      - The calendar window (start/end dates),
      - How month_1..month_6 map to calendar months,
      - That we resolve per-month, not just over the entire window.

    month_labels: optional mapping {1: "December 2025", ..., 6: "May 2026"}.
    """

    _ = hazard_code, metric, resolution_source  # quiet unused-parameter linting

    src = (resolution_source or "").upper()
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    if "ACLED" in src or m == "FATALITIES":
        source_label = "ACLED"
    elif m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        source_label = "IDMC/DTM"
    else:
        source_label = "IFRC Montandon"

    ws = window_start_date.isoformat() if window_start_date else ""
    we = window_end_date.isoformat() if window_end_date else ""

    lines: list[str] = []
    lines.append("TIME HORIZON & RESOLUTION")
    lines.append("")
    if ws or we:
        lines.append(f"- This question covers the period from **{ws}** to **{we}**.")
    else:
        lines.append("- This question covers a six-month period (month_1 to month_6).")

    lines.append("- For scoring, we treat each month separately:")
    if month_labels:
        for idx in range(1, 7):
            label = month_labels.get(idx) or f"month_{idx}"
            lines.append(f"  - `month_{idx}` = {label}")
    else:
        lines.append("  - `month_1` = first calendar month in the forecast window.")
        lines.append("  - `month_2` = second calendar month in the forecast window.")
        lines.append("  - …")
        lines.append("  - `month_6` = sixth calendar month in the forecast window.")

    lines.append(
        "- For each month `m`, Resolver will compute a single monthly value for the "
        f"relevant metric from the underlying source ({source_label}), and "
        "your SPD for that month will be scored against that monthly value."
    )
    lines.append("")
    return "\n".join(lines)

# -------------------------------------------------------------------------------------
# FULL PROMPTS
# -------------------------------------------------------------------------------------

BINARY_PROMPT = _CAL_PREFIX + """
You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign a probability (0–100%) to whether the binary event will occur, using Bayesian reasoning.

Follow these steps in your reasoning before giving the final probability:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate (prior probability P(H)) for the event.
   - Clearly explain why you chose this base rate (e.g., historical frequencies, reference class data, general statistics).
   - State the initial prior in probability or odds form.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference base case.
   - Explain how it is different, and why those differences matter for adjusting the probability.

3. **Evidence Evaluation (Likelihoods)**
   - For each key piece of evidence, consider how likely it would be if the event happens (P(E | H)) versus if it does not happen (P(E | ~H)).
   - Compute or qualitatively describe the likelihood ratio (P(E | H) / P(E | ~H)).
   - State clearly whether each piece of evidence increases or decreases the probability.

4. **Bayesian Updating (Posterior Probability)**
   - Use Bayes’ Rule conceptually:
       Posterior odds = Prior odds × Likelihood ratio
       Posterior probability = (Posterior odds) / (1 + Posterior odds)
   - Walk through at least one explicit update step, showing how the prior probability is adjusted by evidence.
   - Summarize the resulting posterior probability and explain how confident or uncertain it remains.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.
    
5. **Final Forecast**
   - Provide the final forecast as a single calibrated probability.
   - Ensure it reflects both the base rate and the impact of the evidence.

6. **Output Format**
   - End with EXACTLY this line (no other commentary):
Final: ZZ%

Question: {title}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

NUMERIC_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to produce a full probabilistic forecast for a numeric quantity using Bayesian reasoning.

Follow these steps in your reasoning before giving the final percentiles:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate or reference distribution for the target variable.
   - Clearly explain why you chose this base rate (e.g., historical averages, statistical reference classes, domain-specific priors).
   - State the mean/median and variance (or spread) of this base rate.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference distribution.
   - Explain how it is different, and why those differences matter for shifting or stretching the distribution.

3. **Evidence Evaluation (Likelihoods)**
   - For each major piece of evidence in the background or research report, consider how consistent it is with higher vs. lower values.
   - Translate this into a likelihood ratio or qualitative directional adjustment (e.g., “this factor makes higher outcomes 2× as likely as lower outcomes”).
   - Make clear which evidence pushes the forecast up or down, and by how much.

4. **Bayesian Updating (Posterior Distribution)**
   - Use Bayes’ Rule conceptually:
       Posterior ∝ Prior × Likelihood
   - Walk through at least one explicit update step to show how evidence modifies your prior distribution.
   - Describe how the posterior mean, variance, or skew has shifted.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Percentiles**
   - Provide calibrated percentiles that summarize your posterior distribution.
   - Ensure they are internally consistent (P10 < P20 < P40 < P60 < P80 < P90).
   - Think carefully about tail risks and avoid overconfidence.

7. **Output Format**
   - End with EXACTLY these 6 lines (no other commentary):
P10: X
P20: X
P40: X
P60: X
P80: X
P90: X

Question: {title}
Units: {units}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution:
{criteria}

Today (Istanbul time): {today}
"""

MCQ_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign probabilities to each of the multiple-choice options using Bayesian reasoning.
Follow these steps clearly in your reasoning before giving your final answer:

1. **Base Rate (Prior) Selection** - Identify an appropriate base rate (prior probability P(H)) for each option.  
   - Clearly explain why you chose this base rate (e.g., historical frequencies, general statistics, or a reference class).  

2. **Comparison to Base Case** - Explain how the current case is similar to the base rate scenario.  
   - Explain how it is different, and why those differences matter.  

3. **Evidence Evaluation (Likelihoods)** - For each piece of evidence in the background or research report, consider how likely it would be if the option were true (P(E | H)) versus if it were not true (P(E | ~H)).  
   - State these likelihood assessments clearly, even if approximate or qualitative.  

4. **Bayesian Updating (Posterior)** - Use Bayes’ Rule conceptually:  
     Posterior odds = Prior odds × Likelihood ratio  
     Posterior probability = (Posterior odds) / (1 + Posterior odds)  
   - Walk through at least one explicit update step for key evidence, showing how the prior changes into a posterior.  
   - Explain qualitatively how other evidence shifts the probabilities up or down.  

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Normalization** - Ensure the probabilities across all options are consistent and sum to approximately 100%.  
   - Check calibration: if uncertain, distribute probability mass proportionally.  

7. **Output Format** - After reasoning, provide your final forecast as probabilities for each option.  
   - Use EXACTLY N lines, one per option, formatted as:  

Option_1: XX%  
Option_2: XX%  
Option_3: XX%  
...  
(sum ~100%)  

Question: {title}
Options: {options}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

SPD_PROMPT_TEMPLATE = _CAL_PREFIX + """
{scoring_block}
{time_horizon_block}
You are a careful probabilistic forecaster on a humanitarian early warning panel.

Your task is to forecast {quantity_description}.

You will express your beliefs as a SUBJECTIVE PROBABILITY DISTRIBUTION (SPD) over {n_buckets_word} buckets
for each month.

SPD (Subjective Probability Distribution) means:
- You approximate your posterior belief about the monthly value using a small number of discrete buckets.
- Your probabilities over the buckets should reflect the relative plausibility of each range after
  considering the historical base rate and the evidence in the research bundle.

For each month, distribute 100% probability across these buckets:

{bucket_text}

One of these buckets MUST occur for each month. For each month m, your probabilities
{prob_placeholder} must all be between 0 and 1 and sum to approximately 1.0.

Question:
{question}

Background:
{background}

Research bundle (recent/contextual information):
{research}

Resolution criteria (how this metric will be counted):
{resolution_text}

Today (Istanbul time): {today}

---

FORECASTING INSTRUCTIONS (Bayesian SPD)

1) Prior / base rates
   - Start from a prior SPD over the buckets based on historical data and relevant reference classes
     (for this hazard and country).
   - Make your prior explicit in your own thinking: which bucket would you expect *before* reading the evidence?

2) Evidence & likelihood
   - Use the research bundle and history to identify the most important pieces of evidence.
   - For each bucket, ask: “If the true value were in this bucket, how likely is this evidence?”
   - Note which buckets the evidence pushes up or down.

3) Posterior SPD sketch
   - Combine your prior and the evidence qualitatively to sketch a posterior SPD for each month.
   - Check that your SPD:
     - is not implausibly sharp (overconfident), and
     - is not completely flat (ignoring structure).

4) Red-team your forecast
   - Challenge your own forecast:
     - What scenarios might you be underweighting (e.g. rare breakdown of state control, extreme hazard)?
     - Are you systematically underweighting tail risks?
   - Adjust your SPD if needed to reflect realistic but low-probability extreme scenarios.

5) Final JSON output (IMPORTANT)
   - At the very end, output ONLY a single JSON object with this exact schema:

   {{
     "month_1": {prob_placeholder},
     "month_2": {prob_placeholder},
     "month_3": {prob_placeholder},
     "month_4": {prob_placeholder},
     "month_5": {prob_placeholder},
     "month_6": {prob_placeholder}
   }}

   - Do not include any text before or after the JSON.
   - Each list must contain exactly {n_buckets_lower} numbers between 0 and 1 inclusive.
   - For each month, the probabilities must sum to roughly 1.0 (we allow small rounding error).
"""


_BUCKET_COUNT_WORDS = {
    5: ("FIVE", "five"),
    6: ("SIX", "six"),
    7: ("SEVEN", "seven"),
}


def _bucket_count_words(n: int) -> tuple[str, str]:
    """(UPPER, lower) word forms for a bucket count, falling back to digits."""
    return _BUCKET_COUNT_WORDS.get(n, (str(n), str(n)))


def _prob_placeholder(n: int) -> str:
    """Render '[p1, p2, ..., pN]' for the metric's bucket count."""
    return "[" + ", ".join(f"p{i}" for i in range(1, n + 1)) + "]"


def _delta_placeholder(n: int) -> str:
    """Render '[dp1, dp2, ..., dpN]' for the metric's bucket count."""
    return "[" + ", ".join(f"dp{i}" for i in range(1, n + 1)) + "]"


_EXAMPLE_PROBS = {
    5: "0.7,0.2,0.07,0.02,0.01",
    6: "0.55,0.25,0.12,0.05,0.02,0.01",
    7: "0.5,0.25,0.12,0.07,0.03,0.02,0.01",
}


def _example_probs_str(n: int) -> str:
    """A plausible decreasing example probs vector for the bucket count."""
    if n in _EXAMPLE_PROBS:
        return _EXAMPLE_PROBS[n]
    vals = [1.0 / n] * n
    return ",".join(f"{v:.3f}" for v in vals)


SPD_BUCKET_TEXT_PA = """
People affected (PA) buckets (per month, country-level):
- Bucket 1: exactly 0 people affected (label: "0")
- Bucket 2: 1 to < 10,000 people affected (label: "1-<10k")
- Bucket 3: 10,000 to < 50,000 people affected (label: "10k-<50k")
- Bucket 4: 50,000 to < 250,000 people affected (label: "50k-<250k")
- Bucket 5: 250,000 to < 500,000 people affected (label: "250k-<500k")
- Bucket 6: >= 500,000 people affected (label: ">=500k")
"""

SPD_BUCKET_TEXT_FATALITIES = """
Conflict fatalities buckets (per month, country-level):
- Bucket 1: 0 deaths (label: "0")
- Bucket 2: 1–4 deaths (label: "1-<5")
- Bucket 3: 5–24 deaths (label: "5-<25")
- Bucket 4: 25–99 deaths (label: "25-<100")
- Bucket 5: 100–499 deaths (label: "100-<500")
- Bucket 6: 500–999 deaths (label: "500-<1000")
- Bucket 7: >= 1,000 deaths (label: ">=1000")
"""

SPD_BUCKET_TEXT_PHASE3 = """
IPC Phase 3+ population buckets (per month, country-level):
- Bucket 1: exactly 0 people in Phase 3+ (label: "0")
- Bucket 2: 1 to < 100,000 people in Phase 3+ (label: "1-<100k")
- Bucket 3: 100,000 to < 1,000,000 people in Phase 3+ (label: "100k-<1M")
- Bucket 4: 1,000,000 to < 5,000,000 people in Phase 3+ (label: "1M-<5M")
- Bucket 5: 5,000,000 to < 15,000,000 people in Phase 3+ (label: "5M-<15M")
- Bucket 6: >= 15,000,000 people in Phase 3+ (label: ">=15M")
"""

RESEARCHER_PROMPT = """You are a professional RESEARCHER for a Bayesian forecasting panel.
Your job is to produce a concise, decision-useful research brief that helps a statistician
update a prior. The forecasters will combine your brief with a statistical aggregator that
expects: base rates (reference class), recency-weighted evidence (relative to horizon),
key mechanisms, differences vs. the base rate, and indicators to watch. Provide a carefully reasoned, deeply through out research brief. Before answering, lay out for yoursefl your research plan step-by-step. First, identify the core questions to investigate. Second, for each question, propose the search queries you would use. Third, after gathering information, synthesize the key findings. Finally, draft the comprehensive answer."

QUESTION
Title: {title}
Type: {qtype}
Units/Options: {units_or_options}

BACKGROUND
{background}

RESOLUTION CRITERIA (what counts as “true”/resolution)
{criteria}

HORIZON & RECENCY
Today (Istanbul): {today}
Guideline: define “recent” relative to time-to-resolution:
- if >12 months to resolution: emphasize last 24 months
- if 3–12 months: emphasize last 12 months
- if <3 months: emphasize last 6 months

SOURCES (optional; may be empty)
Use these snippets primarily if present; if not present, rely on general knowledge.
Do NOT fabricate precise citations; if unsure, say “uncertain”.
{sources}

=== REQUIRED OUTPUT FORMAT (use headings exactly as written) ===
### Reference class & base rates
- Identify 1–3 plausible reference classes; give ballpark base rates or ranges and reasoning and on how these were derived; note limitations.

### Recent developments (timeline bullets)
- [YYYY-MM-DD] item — direction (↑/↓ for event effect on YES) — why it matters (≤25 words)
- Focus on events within the recency guideline above. Use grounding web search as needed. 

### Mechanisms & drivers (causal levers)
- List 3–6 drivers that move probability up/down; note typical size (small/moderate/large).

### Differences vs. the base rate (what’s unusual now)
- 3–6 bullets contrasting this case with the reference class (structure, actors, constraints, policy).

### Bayesian update sketch (for the statistician)
- Prior: brief sentence suggesting a plausible prior and “equivalent n” (strength).
- Evidence mapping: 3–6 bullets with sign (↑/↓) and rough magnitude (small/moderate/large).
- Net effect: one line describing whether the posterior should move up/down and by how much qualitatively.

### Indicators to watch (leading signals; next weeks/months)
- UP indicators: 3–5 short bullets.
- DOWN indicators: 3–5 short bullets.

### Caveats & pitfalls
- 3–5 bullets on uncertainty, data gaps, deception risks, regime changes, definitional gotchas.

Final Research Summary: One or two sentences for the forecaster. Keep the entire brief under ~3000 words.
"""

# -------------------------------------------------------------------------------------
# BUILDERS
# -------------------------------------------------------------------------------------

def build_binary_prompt(title: str, background: str, research_text: str, criteria: str) -> str:
    return BINARY_PROMPT.format(
        title=title,
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_numeric_prompt(title: str, units: str, background: str, research_text: str, criteria: str) -> str:
    return NUMERIC_PROMPT.format(
        title=title,
        units=(units or "N/A"),
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_mcq_prompt(title: str, options: list[str], background: str, research_text: str, criteria: str) -> str:
    return MCQ_PROMPT.format(
        title=title,
        options="\n".join([str(o) for o in (options or [])]) or "N/A",
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def _format_resolution_text(base_text: str, criteria: str) -> str:
    extra = (criteria or "N/A").strip() or "N/A"
    return f"{base_text}\nAdditional resolution notes: {extra}"


def build_resolution_text_and_quantity_description(
    *,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    resolution_source: Optional[str],
) -> tuple[str, str]:
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    src = (resolution_source or "").upper()

    if m == "FATALITIES" or "ACLED" in src:
        resolution_text = (
            "Fatalities will be measured as deaths recorded by ACLED for this country, "
            "summed over all ACLED event types."
        )
        quantity_description = (
            f"Monthly conflict fatalities in {iso3} ({hazard_label}), all ACLED event "
            "types, as recorded by ACLED."
        )
        return resolution_text, quantity_description

    if m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        resolution_text = (
            "People affected (PA) will be measured as internally displaced people (IDPs), "
            "using IDMC displacement estimates for this hazard and country, with IOM DTM "
            "or comparable humanitarian estimates as fallback."
        )
        quantity_description = (
            f"Monthly internally displaced people (IDPs) in {iso3} due to {hazard_label.lower()}, "
            "as recorded by IDMC and DTM."
        )
        return resolution_text, quantity_description

    resolution_text = (
        "People affected (PA) will be measured using Resolver's canonical PA metric for "
        "this natural hazard and country, based primarily on IFRC Montandon data."
    )
    quantity_description = (
        f"Monthly people affected (PA) in {iso3} by {hazard_label} as recorded by IFRC Montandon "
        "(including both directly and indirectly affected people)."
    )
    return resolution_text, quantity_description


def build_spd_prompt_pa(
    *,
    question_title: str,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    background: str,
    research_text: str,
    resolution_source: Optional[str],
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]],
    today: date,
    criteria: str,
) -> str:
    resolution_text_base, quantity_description = build_resolution_text_and_quantity_description(
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label,
        metric=metric,
        resolution_source=resolution_source,
    )
    resolution_text = _format_resolution_text(resolution_text_base, criteria)

    scoring_block = build_scoring_resolution_block(
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )
    time_horizon_block = build_time_horizon_block(
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )

    today_str = today.isoformat() if isinstance(today, date) else ist_date()

    return SPD_PROMPT_TEMPLATE.format(
        scoring_block=scoring_block,
        time_horizon_block=time_horizon_block,
        question=question_title,
        background=background or "",
        research=research_text or "",
        resolution_text=resolution_text,
        quantity_description=quantity_description,
        bucket_text=SPD_BUCKET_TEXT_PA,
        n_buckets_word=_bucket_count_words(len(labels_for("PA")))[0],
        n_buckets_lower=_bucket_count_words(len(labels_for("PA")))[1],
        prob_placeholder=_prob_placeholder(len(labels_for("PA"))),
        today=today_str,
    )


def build_spd_prompt_fatalities(
    *,
    question_title: str,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    background: str,
    research_text: str,
    resolution_source: Optional[str],
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]],
    today: date,
    criteria: str,
) -> str:
    resolution_text_base, quantity_description = build_resolution_text_and_quantity_description(
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label,
        metric=metric,
        resolution_source=resolution_source,
    )
    resolution_text = _format_resolution_text(resolution_text_base, criteria)

    scoring_block = build_scoring_resolution_block(
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )
    time_horizon_block = build_time_horizon_block(
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )

    today_str = today.isoformat() if isinstance(today, date) else ist_date()

    return SPD_PROMPT_TEMPLATE.format(
        scoring_block=scoring_block,
        time_horizon_block=time_horizon_block,
        question=question_title,
        background=background or "",
        research=research_text or "",
        resolution_text=resolution_text,
        quantity_description=quantity_description,
        bucket_text=SPD_BUCKET_TEXT_FATALITIES,
        n_buckets_word=_bucket_count_words(len(labels_for("FATALITIES")))[0],
        n_buckets_lower=_bucket_count_words(len(labels_for("FATALITIES")))[1],
        prob_placeholder=_prob_placeholder(len(labels_for("FATALITIES"))),
        today=today_str,
    )


def build_spd_prompt(
    *,
    question_title: str,
    background: str,
    research_text: str,
    criteria: str,
    iso3: str = "",
    hazard_code: str = "",
    hazard_label: str = "",
    metric: str = "PA",
    resolution_source: Optional[str] = None,
    window_start_date: Optional[date] = None,
    window_end_date: Optional[date] = None,
    month_labels: Optional[Dict[int, str]] = None,
    today: Optional[date] = None,
) -> str:
    return build_spd_prompt_pa(
        question_title=question_title,
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label or hazard_code,
        metric=metric,
        background=background,
        research_text=research_text,
        resolution_source=resolution_source,
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        today=today or date.today(),
        criteria=criteria,
    )

def build_research_prompt(
    title: str,
    qtype: str,
    units_or_options: str,
    background: str,
    criteria: str,
    today: str,
    sources_text: str,
) -> str:
    """DEPRECATED: The Researcher component has been removed from the pipeline."""
    import warnings
    warnings.warn(
        "build_research_prompt is deprecated - Researcher removed from pipeline",
        DeprecationWarning,
        stacklevel=2,
    )
    sources_text = sources_text.strip() if sources_text else "No external sources provided."
    return RESEARCHER_PROMPT.format(
        title=title,
        qtype=qtype,
        units_or_options=units_or_options or "N/A",
        background=(background or "N/A"),
        criteria=(criteria or "N/A"),
        today=today,
        sources=sources_text,
    )


def _trim_structural(text: str, max_lines: int = 12) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if str(ln or "").strip()]
    return "\n".join(lines[:max_lines]).strip()


def merge_evidence_packs(
    hs_pack: Dict[str, Any] | None,
    question_pack: Dict[str, Any] | None,
    *,
    max_sources: int = 12,
    max_signals: int = 12,
) -> Dict[str, Any]:
    hs_pack = hs_pack or {}
    question_pack = question_pack or {}

    merged_sources = []
    seen_urls: set[str] = set()
    for src in (hs_pack.get("sources") or []) + (question_pack.get("sources") or []):
        if not isinstance(src, dict):
            continue
        url = (src.get("url") or "").strip()
        if url and url not in seen_urls:
            merged_sources.append(src)
            seen_urls.add(url)
        if len(merged_sources) >= max_sources:
            break

    merged_unverified = []
    seen_unverified: set[str] = set(seen_urls)
    for src in (hs_pack.get("unverified_sources") or []) + (question_pack.get("unverified_sources") or []):
        if not isinstance(src, dict):
            continue
        url = (src.get("url") or "").strip()
        if url and url not in seen_unverified:
            merged_unverified.append(src)
            seen_unverified.add(url)
        if len(merged_unverified) >= max_sources:
            break

    hs_struct = _trim_structural(hs_pack.get("structural_context", ""), 8)
    q_struct = _trim_structural(question_pack.get("structural_context", ""), 8)
    structural_context = "\n".join([part for part in (hs_struct, q_struct) if part]).strip()
    if structural_context:
        structural_context = _trim_structural(structural_context, 12)

    combined_signals: list[str] = []
    seen_signals: set[str] = set()
    for sig in (hs_pack.get("recent_signals") or []) + (question_pack.get("recent_signals") or []):
        text = str(sig or "").strip()
        if text and text not in seen_signals:
            combined_signals.append(text)
            seen_signals.add(text)
        if len(combined_signals) >= max_signals:
            break

    grounded = bool(hs_pack.get("grounded")) or bool(question_pack.get("grounded")) or bool(merged_sources)

    return {
        "structural_context": structural_context,
        "recent_signals": combined_signals,
        "sources": merged_sources,
        "unverified_sources": merged_unverified,
        "grounded": grounded,
    }


def build_research_prompt_v2(
    question: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
    resolver_features: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
    question_evidence_pack: Dict[str, Any] | None = None,
    merged_evidence: Dict[str, Any] | None = None,
    prediction_market_bundle: Any | None = None,
) -> str:
    """DEPRECATED: Structured research prompt for Researcher v2.

    The Researcher component has been removed from the pipeline. This function
    is retained for backward compatibility but is no longer called in the main
    forecasting flow.
    """
    import warnings
    warnings.warn(
        "build_research_prompt_v2 is deprecated - Researcher removed from pipeline",
        DeprecationWarning,
        stacklevel=2,
    )

    iso3 = (question.get("iso3") or "").upper()
    hazard = (question.get("hazard_code") or "").upper()
    metric = (question.get("metric") or "").upper()
    resolution_source = question.get("resolution_source", "")
    wording = question.get("wording") or question.get("title") or ""
    model_info = model_info or {}
    rc_scalar_keys = {
        "regime_change_level",
        "regime_change_score",
        "regime_change_likelihood",
        "regime_change_direction",
        "regime_change_magnitude",
        "regime_change_window",
    }
    rc_scalar_present = any(key in hs_triage_entry for key in rc_scalar_keys)

    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    regime_change_level = _coerce_int(hs_triage_entry.get("regime_change_level"))
    regime_change_score = _coerce_float(hs_triage_entry.get("regime_change_score"))
    regime_change_likelihood = _coerce_float(hs_triage_entry.get("regime_change_likelihood"))
    regime_change_direction = hs_triage_entry.get("regime_change_direction")
    regime_change_magnitude = _coerce_float(hs_triage_entry.get("regime_change_magnitude"))
    regime_change_window = hs_triage_entry.get("regime_change_window")

    if not rc_scalar_present:
        rc_payload = hs_triage_entry.get("regime_change")
        if isinstance(rc_payload, dict):
            regime_change_likelihood = _coerce_float(rc_payload.get("likelihood"))
            regime_change_direction = rc_payload.get("direction")
            regime_change_magnitude = _coerce_float(rc_payload.get("magnitude"))
            regime_change_window = rc_payload.get("window")

    rc_elevated = bool(
        (regime_change_level in {2, 3})
        or (regime_change_score is not None and regime_change_score >= 0.30)
    )
    rc_fields_present = any(
        value not in (None, "")
        for value in (
            regime_change_level,
            regime_change_score,
            regime_change_likelihood,
            regime_change_direction,
            regime_change_magnitude,
            regime_change_window,
        )
    )
    rc_level_display = regime_change_level if regime_change_level is not None else "n/a"
    rc_score_display = regime_change_score if regime_change_score is not None else "n/a"
    rc_likelihood_display = regime_change_likelihood if regime_change_likelihood is not None else "n/a"
    rc_direction_display = regime_change_direction if regime_change_direction not in (None, "") else "n/a"
    rc_magnitude_display = regime_change_magnitude if regime_change_magnitude is not None else "n/a"
    rc_window_display = regime_change_window if regime_change_window not in (None, "") else "n/a"

    di_note = ""
    if hazard == "DI":
        di_note = (
            f"For this DI (displacement inflow) hazard in {iso3}, there is **no Resolver base rate**. "
            "You must construct the base rate yourself using UNHCR/IOM flux estimates, known historical incoming "
            "displacement flows from neighbouring countries (e.g., Sudan, Somalia, South Sudan), and appropriate "
            "reference classes. Treat those as the prior, then update it with current signals.\n\n"
            "Do not treat internal conflict displacement within the country as the target metric here; this DI question is "
            "about people entering the country due to events in neighbouring states.\n\n"
        )

    if hazard == "DI":
        base_rate_task = (
            "1. Summarise the base rate of PA for this hazard/country using:\n"
            "   * UNHCR/IOM flow data and documented past inflow episodes from neighbouring countries,\n"
            "   * high-quality external analytical sources (UN, ACAPS, etc.),\n"
            "   * your own knowledge of regional displacement patterns.\n"
        )
    else:
        base_rate_task = (
            f"1. Summarise the base rate of {metric} for this hazard/country using:\n"
            "   * Resolver history (with its caveats),\n"
            "   * high-quality external analytical sources (UN, ACAPS, etc.),\n"
            "   * your own knowledge of the country context.\n"
        )

    merged_pack = merged_evidence or merge_evidence_packs(evidence_pack, question_evidence_pack)

    merged_pack_text = _json_dumps_for_prompt(merged_pack, indent=2)
    parts: list[str] = []
    parts.append("You are a humanitarian risk analyst.")
    parts.append("Your task is to prepare machine-focused research for the forecaster.\n")

    parts.append(
        "Question:\n"
        f"- Country: {iso3}\n"
        f"- Hazard: {hazard}\n"
        f"- Metric: {metric}\n"
        f"- Resolution dataset: {resolution_source}\n"
    )
    rc_block = ["HS REGIME CHANGE FLAG (from HS triage)"]
    if rc_fields_present:
        rc_block.extend(
            [
                f"- RC level: {rc_level_display}",
                f"- RC score: {rc_score_display} (probability × magnitude)",
                f"- RC probability: {rc_likelihood_display}",
                f"- RC direction: {rc_direction_display}",
                f"- RC magnitude: {rc_magnitude_display}",
                f"- RC window: {rc_window_display}",
            ]
        )
    else:
        rc_block.append("(no RC fields provided; treat as unflagged)")
    parts.append("\n".join(rc_block))

    parts.append("Question metadata:\n```json\n" + _json_dumps_for_prompt(question, indent=2) + "\n```")
    parts.append("Natural-language question:\n" + f"\"{wording}\"")
    parts.append(
        "Resolver history (noisy, incomplete base-rate data):\n```json\n"
        + _json_dumps_for_prompt(resolver_features, indent=2)
        + "\n```"
    )
    parts.append(
        "HS triage (tier, triage_score, drivers, regime_shifts, data_quality):\n```json\n"
        + _json_dumps_for_prompt(hs_triage_entry, indent=2)
        + "\n```"
    )

    parts.append(
        "Merged evidence (HS country pack + question-specific web research; prioritize recent signals, structural context is background only):\n"
        "```json\n" + merged_pack_text + "\n```"
    )

    # Render NMME seasonal outlook if present in evidence.
    _seasonal_outlook = (merged_pack or {}).get("nmme_seasonal_outlook")
    if isinstance(_seasonal_outlook, dict) and _seasonal_outlook:
        _seasonal_lines = ["SEASONAL CLIMATE OUTLOOK (NMME multi-model ensemble mean):"]
        for _key, _val in _seasonal_outlook.items():
            _label = _key.replace("_", " ").capitalize()
            _seasonal_lines.append(f"- {_label}: {_val}")
        _seasonal_lines.append(NMME_UNITS_NOTE)
        parts.append("\n".join(_seasonal_lines))

    # Render conflict forecasts for ACE hazard.
    if hazard == "ACE":
        try:
            from horizon_scanner.conflict_forecasts import (
                load_conflict_forecasts,
                format_conflict_forecasts_for_research,
            )
            _cf_data = load_conflict_forecasts(iso3)
            if _cf_data:
                _cf_text = format_conflict_forecasts_for_research(_cf_data)
                if _cf_text:
                    parts.append(_cf_text)
        except Exception:
            pass

    # Render prediction market signals (if available).
    if prediction_market_bundle is not None:
        try:
            pm_text = prediction_market_bundle.to_prompt_text()
            if pm_text:
                parts.append(
                    "PREDICTION MARKET SIGNALS (crowd/market consensus — treat as contextual evidence, not authoritative):\n"
                    "CAVEATS: Markets can be illiquid or manipulated. Manifold uses play money. "
                    "Questions may not directly address this forecast. Absence of markets ≠ absence of risk.\n"
                    + pm_text
                )
        except Exception:
            pass

    parts.append("Model/data notes:\n```json\n" + _json_dumps_for_prompt(model_info, indent=2) + "\n```")

    tasks_block = (
        "Use Resolver as one imperfect signal. ACLED is generally strong for conflict fatalities; IDMC has short history for displacement; IFRC Montandon may be sparse for some hazards/countries; DTM is contextual only.\n"
        + di_note
        + "Your tasks:\n\n"
        + base_rate_task
        + "2. Identify key update signals for the next 6 months that would push risk up or down.\n"
        + "3. Identify specific regime-shift mechanisms that could make the next 6–12 months differ markedly from the past.\n"
        + "4. Note important data gaps and uncertainties.\n\n"
        + "Emphasise major deviations from the historical base rate only when strongly supported by evidence.\n\n"
        + (
            "RC ELEVATED: You MUST include at least 1 item in `regime_shift_signals`. "
            "If you disagree with HS, include a `regime_shift_signals` entry whose description starts with `Rebuttal:` "
            "set likelihood to `low`, and cite sources.\n\n"
            if rc_elevated
            else "`regime_shift_signals` may be empty unless you find credible triggers.\n\n"
        )
        + "Return a single JSON object:\n\n"
        + RESEARCH_V2_REQUIRED_OUTPUT_SCHEMA
        + "\n\nAll URLs in `sources` must be real (no placeholders). If no sources are available, return `sources: []` and `grounded: false`. Set `grounded` to true only when at least one real URL remains after validation. Your `grounded` value will be overridden unless those URLs are verified by the system.\n\n"
        + "Do not include any text outside the JSON."
    )
    parts.append(tasks_block)

    return "\n\n".join(parts) + "\n"

def _bucket_labels_for_question(question: Dict[str, Any]) -> list[str]:
    """Return the metric's bucket labels for this question."""

    metric = (question.get("metric") or "").upper()
    expected_k = n_buckets_for(metric) or n_buckets_for("PA")
    explicit = question.get("bucket_labels") or question.get("class_bins")
    if isinstance(explicit, list) and len(explicit) == expected_k:
        return [str(x) for x in explicit]

    labels = labels_for(metric)
    return list(labels) if labels else labels_for("PA")


def _load_fewsnet_projection(
    iso3: str,
    forecast_keys: list[str],
) -> str:
    """Load FEWS NET Most Likely projection data for a country.

    Queries facts_resolved for metric='phase3plus_projection' rows matching the
    given iso3 and forecast month window.  Returns a formatted table string for
    injection into the SPD prompt, or "" if nothing is found.
    """
    if not iso3 or not forecast_keys:
        return ""
    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            rows = con.execute(
                """
                SELECT ym, value, as_of_date
                FROM facts_resolved
                WHERE iso3 = ?
                  AND metric = 'phase3plus_projection'
                  AND ym >= ?
                  AND ym <= ?
                ORDER BY ym
                """,
                [iso3, forecast_keys[0], forecast_keys[-1]],
            ).fetchall()
        finally:
            duckdb_io.close_db(con)

        if not rows:
            return ""

        lines = [
            "FEWS NET MOST LIKELY PROJECTION (phase3plus_projection):",
            "This is FEWS NET's forward-looking 'Most Likely' scenario for IPC Phase 3+ "
            "population. Use as a moderate-to-strong signal for the direction of change.",
            "",
            "  Month      | Phase 3+ Population | Analysis Date",
            "  -----------|---------------------|-------------",
        ]
        for ym, value, as_of in rows:
            val_str = f"{int(value):,}" if value is not None else "n/a"
            as_of_str = str(as_of) if as_of else "n/a"
            lines.append(f"  {ym:10s} | {val_str:>19s} | {as_of_str}")
        lines.append("")
        return "\n".join(lines)
    except Exception:
        return ""


def _load_haz_base_rate_block(
    iso3: str,
    hazard_code: str,
    metric: str,
    forecast_keys: list[str],
) -> str:
    """Load the PA resolution machine's base-rate block for this question.

    The machine (``resolver/hazard_resolution/``) resolves PA from a detection
    layer plus a fixed reporting ladder, and derives occurrence and severity
    base rates from its own backcast. This injects those rates together with a
    rulebook-generated description of the process — so the forecaster reasons
    about the target it will actually be scored against.

    Eligibility is decided by ``prompt_block.is_eligible``, not here: the rates
    describe people-affected figures for the machine's three hazards, and no
    other metric may be anchored on them. Returns "" whenever there is nothing
    to say — the tables are absent (the machine is not yet wired into a
    workflow), the country-hazard was never assessed, or the load failed.
    Shaped after ``_load_fewsnet_projection`` above, including its connection
    handling.
    """
    if not iso3 or not forecast_keys:
        return ""
    try:
        from resolver.hazard_resolution.prompt_block import (
            build_base_rate_block,
            is_eligible,
        )

        if not is_eligible(hazard_code, metric):
            return ""

        forecast_months: list[int] = []
        for key in forecast_keys:
            match = re.match(r"^\d{4}-(\d{2})", str(key))
            if match:
                forecast_months.append(int(match.group(1)))

        from resolver.db import duckdb_io

        # Env-first, matching duckdb_io._normalize_duckdb_target — the
        # machine's own CLIs (migrate, haz-base-rates) resolve their DB
        # env-first, and a reader that resolves config-first can silently
        # read a different file from the one the writer wrote.
        db_url = os.getenv("RESOLVER_DB_URL", "").strip() or _pythia_db_url_from_config()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            country_name = ""
            try:
                from forecaster.history_loaders import _load_country_names

                country_name = _load_country_names().get((iso3 or "").upper(), "")
            except Exception:
                country_name = ""
            return build_base_rate_block(
                iso3,
                hazard_code,
                metric,
                forecast_months,
                con=con,
                country_name=country_name,
            )
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:
        # Degrade to a prompt without the block, but never silently: this
        # seam is exactly where "loaded but never rendered" failures hide.
        LOG.warning(
            "[prompts] haz base-rate block unavailable for %s/%s/%s: %s",
            iso3, hazard_code, metric, exc,
        )
        return ""


def _forecast_month_keys_from_question(
    question: Dict[str, Any],
    horizon_months: int = 6,
) -> list[str]:
    """
    Derive a list of YYYY-MM month keys for the SPD forecast horizon.

    Thin wrapper over the single-sourced month arithmetic in
    ``forecaster.month_utils``: ``_anchor_month_for_question`` prefers
    ``window_start_date`` (resolutions map horizon_m=1 to that month, so
    the first key MUST be it) and falls back to ``target_months`` /
    ``target_month`` minus 5 (the 6th/last window month per
    create_questions_from_triage); ``_expected_months`` expands the window.
    Returns [] when neither field is usable (caller falls back to generic
    placeholders).
    """
    from forecaster.month_utils import _anchor_month_for_question, _expected_months

    anchor = _anchor_month_for_question(question)
    if not anchor:
        return []
    return _expected_months(anchor, horizon_months)


_HAZARD_EVENT_LABELS = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical cyclone",
}


def _format_gdacs_event_history_for_prompt(
    gdacs_history: Dict[str, Any] | None,
    forecast_months: list[int],
) -> str:
    """Format GDACS event history as a prompt block for SPD and binary forecasts.

    Parameters
    ----------
    gdacs_history
        Dict from _build_gdacs_event_history(), or None.
    forecast_months
        Calendar month numbers (1-12) that the forecast covers.

    Returns
    -------
    str
        Formatted prompt block, or empty string if no data.
    """
    if not gdacs_history:
        return ""

    hz = gdacs_history.get("hazard_code", "")
    iso3 = gdacs_history.get("iso3", "")
    event_label = _HAZARD_EVENT_LABELS.get(hz, hz)
    data_range = gdacs_history.get("data_range", "")
    total = gdacs_history.get("total_months", 0)
    event_months = gdacs_history.get("event_months", 0)
    event_rate = gdacs_history.get("event_rate_pct", 0)
    alert_dist = gdacs_history.get("alert_distribution", {})
    seasonal = gdacs_history.get("seasonal", {})
    recent_12 = gdacs_history.get("recent_12", [])

    _MONTH_NAMES = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    lines: list[str] = []
    lines.append(
        f"GDACS EVENT HISTORY ({iso3} — {event_label}):"
    )
    lines.append("Source: GDACS (Global Disaster Alert and Coordination System)")
    if gdacs_history.get("history_available") is False:
        # A handful of months is not a history; printing its rate would
        # read as a confident 0% (see forecaster/gdacs_history.py).
        from forecaster.gdacs_history import unavailable_line

        lines.append(unavailable_line(
            event_label, iso3,
            gdacs_history.get("unavailable_reason") or "too few months of GDACS coverage",
        ))
        return "\n".join(lines)
    lines.append(f"Coverage: {data_range} ({total} calendar months; a month with no alert counts as no event)")
    lines.append(
        f"Overall event rate: {event_months} events in {total} months "
        f"({event_rate:.0f}%)"
    )

    # Alert distribution
    if alert_dist:
        dist_parts = [f"{level}: {count}" for level, count in sorted(alert_dist.items())]
        lines.append(f"Alert levels (event months only): {', '.join(dist_parts)}")

    # Seasonal frequency for forecast months
    lines.append("")
    lines.append("Seasonal event frequency (your forecast months):")
    for cal_month in forecast_months:
        s = seasonal.get(cal_month) or seasonal.get(str(cal_month)) or {}
        years_obs = s.get("years_observed", 0)
        years_evt = s.get("years_with_event", 0)
        freq = s.get("frequency_pct", 0)
        month_name = _MONTH_NAMES.get(cal_month, str(cal_month))
        if years_obs > 0:
            lines.append(
                f"  {month_name}: {years_evt}/{years_obs} years had events ({freq:.0f}%)"
            )
        else:
            lines.append(f"  {month_name}: no historical data")

    # Recent 12 months timeline
    if recent_12:
        lines.append("")
        lines.append("Recent event timeline:")
        # Format in rows of 3
        for i in range(0, len(recent_12), 3):
            chunk = recent_12[i:i + 3]
            parts = []
            for entry in chunk:
                ym = entry.get("ym", "")
                if entry.get("occurred"):
                    level = entry.get("alertlevel") or "?"
                    parts.append(f"{ym}: {level}")
                else:
                    parts.append(f"{ym}: --")
            lines.append("  " + " | ".join(parts))

    lines.append("")
    lines.append(
        "INTERPRETATION: GDACS events indicate confirmed disaster activity "
        "(Orange = moderate, Red = severe). Months with no event (--) mean "
        "GDACS detected no significant disaster of this type. Use event "
        "frequency as a prior for whether impact will be zero vs non-zero "
        "in each forecast month. When an event occurs, use the IFRC/Resolver "
        "PA seasonal profile to estimate magnitude."
    )

    return "\n".join(lines)


def _build_base_rate_text(
    history_summary: Dict[str, Any],
    forecast_keys: list[str],
    iso3: str,
    hazard_code: str,
    metric: str = "",
) -> str:
    """Lazy-import wrapper to avoid circular imports between prompts and cli."""
    from .cli import _format_base_rate_for_prompt  # noqa: F811 — deferred to avoid cycle

    return _format_base_rate_for_prompt(
        history_summary, forecast_keys, iso3=iso3, hazard_code=hazard_code, metric=metric
    )


def _seasonal_profile_has_observations(history_summary: Dict[str, Any]) -> bool:
    months = (history_summary or {}).get("months") or {}
    for m_data in months.values():
        try:
            if int((m_data or {}).get("n_observations") or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _one_natural_hazard_anchor(
    base_rate_text: str,
    history_summary: Dict[str, Any],
    has_machine_block: bool,
    iso3: str,
    hazard: str,
) -> str:
    """One base-rate anchor per FL/TC PA prompt (Oct 2026).

    The legacy per-calendar-month IFRC profile printed beside the PA
    machine's backcast block, so the model was handed two priors built two
    ways, and where IFRC had no rows it printed "IFRC data from  (0 years)"
    above a table of zeros (13 of 22 FL/TC PA prompts of the 1 Oct 2026
    run). The machine's block wins where it is present; an empty profile is
    replaced by one line saying the record is empty.
    """
    if (history_summary or {}).get("type") != "seasonal_profile":
        return base_rate_text
    if has_machine_block:
        return (
            "RESOLVER HISTORY: the base rate for this question is the PA "
            "resolution machine's block below; the IFRC-only profile is not "
            "shown beside it."
        )
    if not _seasonal_profile_has_observations(history_summary):
        return (
            f"BASE RATE: no reported people-affected figure for {iso3} "
            f"{hazard} in the Resolver record. This is an absence of reports, "
            "not evidence that nobody was affected; build your prior from the "
            "seasonal and structured evidence below."
        )
    return base_rate_text


def _prompt_v3_order_enabled() -> bool:
    """Static-first prompt section ordering (PYTHIA_PROMPT_V3_ORDER).

    Legacy order puts per-question data first and the big static method/schema
    blocks last, which caps the cacheable prompt prefix at ~84 bytes. V3 order
    puts the static blocks first so requests within a (hazard, metric, track)
    group share a multi-KB prefix — the enabler for OpenAI automatic caching,
    Gemini implicit caching, and Anthropic cache_control. Section TEXT is
    identical in both orders (only positional referents like "above"/"below"
    change); default OFF until the test-mode A/B validation run passes.
    """

    return os.getenv("PYTHIA_PROMPT_V3_ORDER", "0").strip().lower() in ("1", "true", "yes")


# --- Regime-change SHIFT guidance (PYTHIA_RC_SHIFT_GUIDANCE) -------------------
# The legacy RC guidance asks members to WIDEN the posterior when HS flags a
# regime change. Country-month conflict deaths are persistent, so a widened
# distribution loses to a sharp base-rate anchor; the shift guidance asks the
# member to MOVE mass in the flagged direction and keep the prior's shape.
# Off by default: with the flag off every prompt is byte-identical to before.
RC_SHIFT_GUIDANCE_VERSION = "shift_v1"

# Static text (identical for every Track 1 question while the flag is on, so it
# lives in the cacheable prefix; the per-question numbers live in the data tail).
_RC_SHIFT_METHOD_BLOCK = (
    "REGIME-CHANGE FLAG: MOVE THE DISTRIBUTION, DO NOT WIDEN IT (months 1-3)\n"
    "When the question data carries an HS regime-change flag at level 1 or above:\n"
    "  - Start from your Step 1 base-rate prior. A flag is a claim about the DIRECTION the outcome "
    "is heading, not a reason to be less sure of everything.\n"
    "  - If you accept the flag, shift probability mass toward the flagged direction by an amount "
    "that matches the flag's stated likelihood and magnitude. Keep the distribution about as "
    "concentrated as the prior unless the evidence is genuinely two-sided.\n"
    "  - UP moves mass to higher buckets and DOWN to lower ones; neither adds mass on the "
    "opposite side of the prior's modal bucket. Only a mixed or unclear flag justifies adding "
    "mass to both tails, and then only modestly.\n"
    "  - Honour the sharpness anchor stated in the question data, or explain in "
    "`reasoning_trace.rc_shift.why` why the evidence overrides it.\n"
    "  - This is about the SHAPE of months 1-3 under a flag. Any widening of later months "
    "(months 4-6) for growing uncertainty is a separate matter and still applies.\n"
    "  - If you rebut the flag, keep the prior's shape and say so.\n\n"
)
_RC_SHIFT_TRACE_INSTRUCTION = (
    "- `reasoning_trace.rc_shift` states how the regime-change flag moved your month-1 SPD: "
    "`direction` (up, down, none or two_sided), `expected_bucket_change` (posterior minus prior "
    "expected bucket index, signed), `mass_moved` (share of probability moved, 0 to 1), "
    "`sharpness_kept` (true if the sharpness anchor holds) and `why` (one sentence).\n"
)
_RC_SHIFT_SCHEMA_LINES = (
    '    "rc_shift": {"direction": "up or down or none or two_sided", '
    '"expected_bucket_change": 0.4, "mass_moved": 0.15, "sharpness_kept": true, '
    '"why": "one sentence"}\n'
)


def rc_shift_guidance_enabled() -> bool:
    """PYTHIA_RC_SHIFT_GUIDANCE (0/1, default 0)."""

    return os.getenv("PYTHIA_RC_SHIFT_GUIDANCE", "0").strip().lower() in ("1", "true", "yes")


def rc_guidance_version(track: int = 1) -> Optional[str]:
    """The RC guidance a member prompt of this track carries, for ``forecasts_raw.rc_guidance``.

    ``None`` means the legacy wording. Only Track 1 prompts change.
    """

    if track < 2 and rc_shift_guidance_enabled():
        return RC_SHIFT_GUIDANCE_VERSION
    return None


def _rc_direction_kind(direction: Any) -> str:
    """'up' | 'down' | 'two_sided' (mixed, unclear or absent)."""

    d = str(direction or "").strip().lower()
    if d in ("up", "down"):
        return d
    return "two_sided"


def _load_base_rate_modal(question: Dict[str, Any]) -> Optional[tuple]:
    """``(modal_index, modal_prob, probs)`` of the base-rate SPD, or None.

    The same anchor ``score_baselines`` scores as climatology
    (``pythia.tools.base_rate_spd``). Called only when the shift guidance is
    on, so the default path touches no database. Degrades to None, never raises.
    """

    iso3 = (question.get("iso3") or "").upper()
    hazard = (question.get("hazard_code") or "").upper()
    metric = (question.get("metric") or "").upper()
    as_of = question.get("window_start_date")
    if not iso3 or not hazard or not metric or not as_of:
        return None
    try:
        from pythia.tools.base_rate_spd import base_rate_spd
        from resolver.db import duckdb_io

        db_url = os.getenv("RESOLVER_DB_URL", "").strip() or _pythia_db_url_from_config()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            probs, _source, _detail = base_rate_spd(con, iso3, hazard, metric, as_of)
        finally:
            duckdb_io.close_db(con)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[prompts] base-rate anchor unavailable for %s/%s/%s: %s", iso3, hazard, metric, exc)
        return None
    if not probs or len(probs) < 2:
        return None
    modal = max(range(len(probs)), key=lambda i: probs[i])
    return modal, float(probs[modal]), list(probs)


def _rc_shift_question_guidance(
    *,
    buckets: list[str],
    rc_dir_display: Any,
    rc_prob_display: Any,
    rc_mag_display: Any,
    base_rate_modal: Optional[tuple],
) -> str:
    """The per-question half of the shift guidance: this flag, this anchor."""

    kind = _rc_direction_kind(rc_dir_display)
    lines = [
        "How to use this flag in months 1-3 (see REGIME-CHANGE FLAG in the method):",
        "- Start from the base-rate prior. If you accept the flag, MOVE probability mass "
        f"toward the flagged direction by an amount that matches the stated likelihood "
        f"({rc_prob_display}) and magnitude ({rc_mag_display}). Keep the distribution about "
        "as concentrated as the prior unless the evidence is genuinely two-sided.",
    ]
    if kind == "up":
        lines.append(
            "- Direction UP: move mass to higher buckets, taking it from the lower buckets. "
            "Do not add mass to buckets below the prior's modal bucket."
        )
    elif kind == "down":
        lines.append(
            "- Direction DOWN: move mass to lower buckets, taking it from the higher buckets. "
            "Do not add mass to buckets above the prior's modal bucket."
        )
    else:
        lines.append(
            "- Direction mixed or unclear: this is the only case that justifies adding mass to "
            "both tails, and then only modestly; the modal bucket should stay the modal bucket."
        )
    if base_rate_modal is not None and buckets:
        modal, modal_p, _probs = base_rate_modal
        if 0 <= modal < len(buckets):
            if kind == "up":
                nb = min(modal + 1, len(buckets) - 1)
            elif kind == "down":
                nb = max(modal - 1, 0)
            else:
                nb = None
            if nb is not None and nb != modal:
                keep = f'the "{buckets[modal]}" and "{buckets[nb]}" buckets together'
            else:
                keep = f'the "{buckets[modal]}" bucket and its immediate neighbours together'
            lines.append(
                f'- Sharpness anchor: the base-rate distribution puts {modal_p:.0%} on its modal '
                f'bucket "{buckets[modal]}". Your month-1 posterior must keep at least {modal_p:.0%} '
                f"on {keep}, unless you explain why not in `reasoning_trace.rc_shift.why`."
            )
    else:
        lines.append(
            "- Sharpness anchor: your month-1 posterior must keep at least your Step 1 prior's "
            "modal-bucket probability on that bucket plus its neighbour in the flagged direction, "
            "unless you explain why not in `reasoning_trace.rc_shift.why`."
        )
    return "\n".join(lines) + "\n\n"


# --- Base-rate distribution as the prior (PYTHIA_PRIOR_ANCHOR_SPD) ------------
# On the resolved August 2026 conflict-death questions the members' priors put
# 0.38 on the realised bucket where the climatology SPD put 0.54, and the loss
# was in that prior, not in the update. The flag hands ACE/FATALITIES members
# the level-and-volatility distribution (base_rate_spd.level_volatility_spds)
# and tells them to copy it as the prior, so every departure has to be argued
# as an update. Off by default: with the flag off every prompt is
# byte-identical to before.

#: Rendered block over this many characters is logged, and a test fails; the
#: block is never cut, because a distribution cut off mid-bucket still reads
#: as a complete one.
PRIOR_ANCHOR_MAX_CHARS = 1400

_PRIOR_ANCHOR_STEP1 = (
    "Your prior MUST be the BASE-RATE DISTRIBUTION given {ref}, copied exactly: month 1 and "
    "month 6 as printed, months 2 to 5 interpolated linearly between them. Do not adjust it "
    "here. Every departure from it is an update: argue it in STEP 3 and record it in "
    "`updates[]` with its delta.\n"
)


def prior_anchor_enabled() -> bool:
    """True when ``PYTHIA_PRIOR_ANCHOR_SPD`` is on (default off)."""
    return os.getenv("PYTHIA_PRIOR_ANCHOR_SPD", "0").strip().lower() in ("1", "true", "yes")


def prior_anchor_version() -> str:
    """The block wording in force: ``prior_anchor_v1`` unless
    ``PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION`` is ``v2`` (or ``prior_anchor_v2``).

    The distribution is the same under both; only the Spread sentence moves.
    v1 stays the default until the October 2026 run (the first v1 run) has
    been scored, because a correction is fitted per block version and two
    wordings in one month would split that month's evidence.
    """
    from pythia.tools.base_rate_spd import (
        LEVEL_VOLATILITY_VERSION,
        LEVEL_VOLATILITY_VERSION_V2,
    )

    raw = os.getenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", "").strip().lower()
    if raw in ("v2", "2", LEVEL_VOLATILITY_VERSION_V2):
        return LEVEL_VOLATILITY_VERSION_V2
    return LEVEL_VOLATILITY_VERSION


@functools.lru_cache(maxsize=512)
def _prior_anchor_cached(db_url: str, iso3: str, window_ym: str, today_iso: str):
    from pythia.tools.base_rate_spd import level_volatility_spds
    from resolver.db import duckdb_io

    con = duckdb_io.get_db(db_url)
    try:
        return level_volatility_spds(con, iso3, window_ym, known_at=today_iso)
    finally:
        duckdb_io.close_db(con)


def load_prior_anchor(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The level-and-volatility distribution for an ACE/FATALITIES question, or None.

    None when the flag is off, the question is another hazard or metric, or
    there is no complete ACLED month to anchor on. Never raises.
    """

    if not prior_anchor_enabled():
        return None
    hazard = (question.get("hazard_code") or "").upper()
    metric = (question.get("metric") or "").upper()
    if hazard != "ACE" or metric != "FATALITIES":
        return None
    iso3 = (question.get("iso3") or "").upper()
    window = str(question.get("window_start_date") or "")[:7]
    if not iso3 or len(window) != 7:
        return None
    try:
        from resolver.db import duckdb_io

        db_url = os.getenv("RESOLVER_DB_URL", "").strip() or _pythia_db_url_from_config()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        today = os.getenv("PYTHIA_PRIOR_ANCHOR_TODAY", "").strip() or date.today().isoformat()
        spds, source, detail = _prior_anchor_cached(db_url, iso3, window, today)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[prompts] prior anchor unavailable for %s: %s", iso3, exc)
        return None
    if 1 not in spds or 6 not in spds:
        if detail.get("reason"):
            LOG.info("[prompts] no prior anchor for %s: %s", iso3, detail.get("reason"))
        return None
    return {"version": prior_anchor_version(), "spds": spds, "source": source, "detail": detail}


def prior_anchor_block_version(question: Dict[str, Any]) -> Optional[str]:
    """What ``forecasts_raw.base_rate_block_version`` records for this question."""
    anchor = load_prior_anchor(question)
    return anchor["version"] if anchor else None


def _pct(p: float) -> str:
    v = 100.0 * float(p)
    return f"{v:.1f}%" if v < 1 else f"{v:.0f}%"


def _prior_anchor_v2_name() -> str:
    from pythia.tools.base_rate_spd import LEVEL_VOLATILITY_VERSION_V2

    return LEVEL_VOLATILITY_VERSION_V2


def render_prior_anchor_block(anchor: Dict[str, Any], forecast_keys: list[str]) -> str:
    """The BASE-RATE DISTRIBUTION block (per-question data; after the cache prefix)."""
    from pythia.buckets import labels_for

    labels = labels_for("FATALITIES")
    detail = anchor["detail"]
    h1 = (detail.get("horizons") or {}).get("1") or {}
    level_ym = str(detail.get("level_month") or "")
    try:
        level_name = datetime.strptime(level_ym, "%Y-%m").strftime("%B %Y")
    except ValueError:
        level_name = level_ym
    level_value = int(round(float(detail.get("level_value") or 0)))
    level_label = labels[int(detail.get("level_bucket") or 0)]
    gap = int(h1.get("gap_months") or 1)
    later = "one month later" if gap == 1 else f"{gap} months later"
    n_months = int(detail.get("n_months_in_window") or 0)
    if anchor.get("version") == _prior_anchor_v2_name():
        # v2 reads the Spread off the vectors printed below, so the sentence
        # and the numbers a member copies cannot disagree. The v1 shares are
        # of raw moves, before mass past either end is clipped onto the end
        # bucket and before the floor, so at bucket 0 or the top bucket they
        # understate "stays".
        lb = int(detail.get("level_bucket") or 0)

        def _split(vec: list) -> tuple:
            return (sum(vec[lb + 1:]), float(vec[lb]), sum(vec[:lb]))

        up1, stay1, down1 = _split(anchor["spds"][1])
        up6, stay6, down6 = _split(anchor["spds"][6])
        sentence = (
            f"Read off the distribution below: in month 1 the count stays in bucket "
            f"{level_label} with probability {_pct(stay1)}, moves to a higher bucket "
            f"{_pct(up1)} and to a lower one {_pct(down1)}; by month 6, "
            f"{_pct(stay6)}, {_pct(up6)} and {_pct(down6)}. These come from how far "
            f"counts moved over the same gap in the last {n_months} complete months."
        )
        if h1.get("pooled"):
            sentence += (
                f" This country's own record gives only {int(h1.get('n_own_pairs') or 0)} "
                f"such pairs, so the moves pool {int(h1.get('n_band_countries') or 0)} "
                f"countries whose typical month falls in the same bucket."
            )
    else:
        sentence = (
            f"Over the last {n_months} complete months, the count {later} stayed in the same "
            f"bucket {_pct(h1.get('share_same', 0))} of the time, moved one bucket "
            f"{_pct(h1.get('share_one', 0))}, and two or more {_pct(h1.get('share_two_plus', 0))}."
        )
        if h1.get("pooled"):
            sentence += (
                f" This country's own record gives only {int(h1.get('n_own_pairs') or 0)} such "
                f"pairs, so the shares pool {int(h1.get('n_band_countries') or 0)} countries whose "
                f"typical month falls in the same bucket."
            )

    def _row(h: int) -> str:
        probs = anchor["spds"][h]
        key = forecast_keys[h - 1] if len(forecast_keys) >= h else f"month {h}"
        cells = "; ".join(f"{lab} {_pct(p)}" for lab, p in zip(labels, probs))
        return f"  Month {h} ({key}): {cells}"

    block = (
        "BASE-RATE DISTRIBUTION (your Step 1 prior):\n"
        f"  Level: {level_value:,} deaths in {level_name} (bucket {level_label}), "
        "the last complete month in the record.\n"
        f"  Spread: {sentence}\n"
        f"{_row(1)}\n"
        f"{_row(6)}\n"
        "  Months 2 to 5: interpolate linearly between month 1 and month 6.\n"
    )
    if len(block) > PRIOR_ANCHOR_MAX_CHARS:
        LOG.warning(
            "[prompts] prior anchor block is %d chars, over the %d budget (not truncated)",
            len(block), PRIOR_ANCHOR_MAX_CHARS,
        )
    return block


def build_spd_prompt_v2(
    question: Dict[str, Any],
    history_summary: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
    research_json: Dict[str, Any],
    structured_data: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    track: int = 1,
    return_parts: bool = False,
):
    """Assemble the SPD v2 forecasting prompt with structured context.

    research_json is now a minimal dict containing only prediction market
    signals, NMME seasonal outlook, and/or hazard tail pack data. The full
    research narrative is no longer produced (Researcher component removed).

    With ``return_parts=True`` returns ``(stable_prefix, dynamic_suffix)``
    (joined they equal the normal return value). The prefix is byte-identical
    across all questions of the same (hazard, metric, track) group under V3
    order, so callers can hang provider cache markers on it; under legacy
    order the prefix is "" (nothing usefully cacheable — skip caching).
    """

    iso3 = (question.get("iso3") or "").upper()
    hazard = (question.get("hazard_code") or "").upper()
    metric = (question.get("metric") or "").upper()
    resolution_source = (question.get("resolution_source") or "").upper()
    wording = question.get("wording") or question.get("title") or ""

    # Load hazard-specific calibration advice
    cal_advice_text = ""
    if advice_arm(question.get("question_id")) != "no_advice":
        cal_advice_text = _load_calibration_advice_for_hazard(hazard, metric, model_name=model_name)
    calibration_section = ""
    if cal_advice_text:
        calibration_section = (
            "CALIBRATION GUIDANCE (auto-generated from historical scoring):\n"
            + cal_advice_text
            + "\n--- end calibration ---\n\n"
        )

    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    rc_level = _coerce_int(hs_triage_entry.get("regime_change_level"))
    rc_score = _coerce_float(hs_triage_entry.get("regime_change_score"))
    rc_prob = _coerce_float(hs_triage_entry.get("regime_change_likelihood"))
    rc_dir = hs_triage_entry.get("regime_change_direction")
    rc_mag = _coerce_float(hs_triage_entry.get("regime_change_magnitude"))
    rc_window = hs_triage_entry.get("regime_change_window")

    rc_scalar_present = any(
        key in hs_triage_entry
        for key in (
            "regime_change_level",
            "regime_change_score",
            "regime_change_likelihood",
            "regime_change_direction",
            "regime_change_magnitude",
            "regime_change_window",
        )
    )

    if not rc_scalar_present:
        rc_payload = hs_triage_entry.get("regime_change")
        if isinstance(rc_payload, dict):
            rc_prob = _coerce_float(rc_payload.get("likelihood"))
            rc_dir = rc_payload.get("direction")
            rc_mag = _coerce_float(rc_payload.get("magnitude"))
            rc_window = rc_payload.get("window")

    if rc_level is not None:
        rc_level_effective = rc_level
    elif rc_score is not None:
        if rc_score >= 0.45:
            rc_level_effective = 3
        elif rc_score >= 0.30:
            rc_level_effective = 2
        elif rc_score >= 0.20 and rc_prob is not None and rc_prob >= 0.35:
            rc_level_effective = 1
        else:
            rc_level_effective = 0
    else:
        rc_level_effective = 0

    rc_level_display = rc_level if rc_level is not None else "n/a"
    rc_score_display = rc_score if rc_score is not None else "n/a"
    rc_prob_display = rc_prob if rc_prob is not None else "n/a"
    rc_dir_display = rc_dir if rc_dir not in (None, "") else "n/a"
    rc_mag_display = rc_mag if rc_mag is not None else "n/a"
    rc_window_display = rc_window if rc_window not in (None, "") else "n/a"

    forecast_keys = _forecast_month_keys_from_question(question, horizon_months=6)
    forecast_labels: list[str] = []
    if forecast_keys:
        for k in forecast_keys:
            try:
                y = int(k[0:4])
                m = int(k[5:7])
                d = date(y, m, 1)
                forecast_labels.append(d.strftime("%B %Y"))
            except Exception:
                forecast_labels.append(k)

    base_rate_note = ""
    if (history_summary.get("source") or "").lower() == "none":
        base_rate_note = (
            "- Resolver does not currently provide a base-rate series for this hazard; treat the base rate as uncertain and lean more heavily on HS triage + research.\n"
        )

    buckets = _bucket_labels_for_question(question)
    if metric == "FATALITIES" or "ACLED" in resolution_source:
        unit_phrase = "people killed (conflict fatalities) per month"
    elif metric == "PHASE3PLUS_IN_NEED":
        unit_phrase = "people in IPC Phase 3+ (crisis or worse) per month"
    else:
        unit_phrase = "people affected or displaced per month"
    n_buckets = len(buckets)
    n_buckets_lower = _bucket_count_words(n_buckets)[1]
    prob_ph = _prob_placeholder(n_buckets)
    delta_ph = _delta_placeholder(n_buckets)
    example_probs = _example_probs_str(n_buckets)

    hazard_reasoning_block = get_hazard_reasoning_block(hazard, metric)

    bucket_list_str = ", ".join([f'\"{b}\"' for b in buckets])

    if forecast_keys and len(forecast_keys) >= 2:
        example_key_1 = forecast_keys[0]
        example_key_2 = forecast_keys[1]
    else:
        example_key_1 = "YYYY-MM"
        example_key_2 = "YYYY-MM+1"

    horizon_note = ""
    if forecast_keys and forecast_labels and len(forecast_keys) == len(forecast_labels) == 6:
        horizon_note = (
            "Forecast horizon (months and JSON keys):\n"
            f"- Month 1: {forecast_labels[0]} (key: \"{forecast_keys[0]}\")\n"
            f"- Month 2: {forecast_labels[1]} (key: \"{forecast_keys[1]}\")\n"
            f"- Month 3: {forecast_labels[2]} (key: \"{forecast_keys[2]}\")\n"
            f"- Month 4: {forecast_labels[3]} (key: \"{forecast_keys[3]}\")\n"
            f"- Month 5: {forecast_labels[4]} (key: \"{forecast_keys[4]}\")\n"
            f"- Month 6: {forecast_labels[5]} (key: \"{forecast_keys[5]}\")\n\n"
            "You MUST use exactly these six keys in the `spds` object (no extra months).\n\n"
        )

    rc_guidance = (
        "REGIME CHANGE GUIDANCE (RC):\n"
        f"- RC level: {rc_level_display}\n"
        f"- RC score: {rc_score_display}\n"
        f"- RC probability: {rc_prob_display}\n"
        f"- RC direction: {rc_dir_display}\n"
        f"- RC magnitude: {rc_mag_display}\n"
        f"- RC window: {rc_window_display}\n"
        "Guidance by RC level:\n"
        "- Level 0: base-rate normal; still consider tails.\n"
        "- Level 1: sanity-check base-rate anchoring; consider modest tail widening.\n"
        "- Level 2: treat base rate as less reliable; widen posterior; ensure non-trivial tail mass in the RC direction unless rebutted.\n"
        "- Level 3: explicitly model a regime-shift scenario; avoid narrow SPDs; tails must be meaningfully represented if direction is UP/DOWN.\n\n"
    )
    rc_shift_on = rc_guidance_version(track) is not None
    if rc_shift_on:
        # Same header and flag values; the legacy "widen" lines are replaced.
        rc_guidance = (
            "REGIME CHANGE GUIDANCE (RC):\n"
            f"- RC level: {rc_level_display}\n"
            f"- RC score: {rc_score_display}\n"
            f"- RC probability: {rc_prob_display}\n"
            f"- RC direction: {rc_dir_display}\n"
            f"- RC magnitude: {rc_mag_display}\n"
            f"- RC window: {rc_window_display}\n"
        )
        if rc_level_effective >= 1:
            rc_guidance += _rc_shift_question_guidance(
                buckets=_bucket_labels_for_question(question),
                rc_dir_display=rc_dir_display,
                rc_prob_display=rc_prob_display,
                rc_mag_display=rc_mag_display,
                base_rate_modal=_load_base_rate_modal(question),
            )
        else:
            rc_guidance += "- No regime change flagged: stay with the base rate; still consider tails.\n\n"
    rc_self_search_line = ""
    if rc_level_effective >= 2 and _self_search_enabled():
        rc_self_search_line = (
            "If sources/signals are sparse in the merged evidence, you may respond with:\n"
            f"NEED_WEB_EVIDENCE: {iso3} {hazard} leading indicators next 3 months escalation trigger OR de-escalation trigger humanitarian\n\n"
        )

    pm_signals_section = ""
    _pm_signals = research_json.get("prediction_market_signals")
    if isinstance(_pm_signals, dict) and _pm_signals.get("questions"):
        pm_signals_section = (
            "PREDICTION MARKET SIGNALS:\n"
            "The research brief includes prediction market signals. "
            "If relevant, consider them as additional base-rate anchors. "
            "Weight by platform (Metaculus > Polymarket > Manifold) and liquidity.\n\n"
        )

    seasonal_outlook_section = ""
    _seasonal_data = research_json.get("nmme_seasonal_outlook")
    if isinstance(_seasonal_data, dict) and _seasonal_data:
        _lines = ["SEASONAL CLIMATE OUTLOOK (NMME multi-model ensemble mean):"]
        for _k, _v in _seasonal_data.items():
            _label = _k.replace("_", " ").capitalize()
            _lines.append(f"- {_label}: {_v}")
        _lines.append(NMME_UNITS_NOTE + "\n")
        seasonal_outlook_section = "\n".join(_lines) + "\n"

    # --- NEW: Structured data sections from connectors ---
    structured_sections: list[str] = []
    sd = structured_data or {}

    if sd.get("reliefweb_reports"):
        try:
            from horizon_scanner.reliefweb import format_reliefweb_for_spd
            t = format_reliefweb_for_spd(sd["reliefweb_reports"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("acled_political_events") and hazard in ("ACE", "DI"):
        try:
            from pythia.acled_political import format_political_events_for_spd
            t = format_political_events_for_spd(sd["acled_political_events"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("ipc_phases"):
        try:
            from pythia.ipc_phases import format_ipc_for_spd
            t = format_ipc_for_spd(sd["ipc_phases"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("fewsnet_food_security"):
        try:
            from pythia.food_security import format_food_security_for_spd
            t = format_food_security_for_spd(sd["fewsnet_food_security"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("inform_severity"):
        try:
            from pythia.acaps import format_inform_severity_for_spd
            t = format_inform_severity_for_spd(sd["inform_severity"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("acaps_risk_radar"):
        try:
            from pythia.acaps import format_risk_radar_for_spd
            t = format_risk_radar_for_spd(sd["acaps_risk_radar"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("acaps_monitoring"):
        try:
            from pythia.acaps import format_daily_monitoring_for_spd
            t = format_daily_monitoring_for_spd(sd["acaps_monitoring"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("adversarial_check") and rc_level_effective >= 1:
        try:
            from pythia.adversarial_check import format_adversarial_check_for_spd
            t = format_adversarial_check_for_spd(sd["adversarial_check"], rc_level=rc_level_effective)
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    if sd.get("crisiswatch"):
        # Pre-formatted by horizon_scanner.crisiswatch.format_crisiswatch_for_prompt
        # (loaded for ACE questions only in _load_structured_data).
        structured_sections.append(str(sd["crisiswatch"]))

    if sd.get("hazard_grounding"):
        _hg = sd["hazard_grounding"]
        # RC/triage grounding packs store their text under `report_markdown`
        # (see _load_hs_hazard_tail_pack); accept the legacy `markdown` key too.
        _hg_md = None
        if isinstance(_hg, dict):
            _hg_md = _hg.get("report_markdown") or _hg.get("markdown")
        if _hg_md:
            structured_sections.append("HS GROUNDING EVIDENCE:\n" + str(_hg_md))

    if sd.get("conflict_forecasts") and hazard in ("ACE", "DI"):
        structured_sections.append(sd["conflict_forecasts"])

    if sd.get("gdelt_conflict_indicators") and hazard == "ACE":
        structured_sections.append(sd["gdelt_conflict_indicators"])

    if sd.get("hdx_signals"):
        structured_sections.append(sd["hdx_signals"])

    if sd.get("enso_context") and hazard in ("TC", "FL", "DR", "HW"):
        structured_sections.append(sd["enso_context"])

    if sd.get("seasonal_tc_context") and hazard == "TC":
        try:
            from horizon_scanner.seasonal_tc import format_seasonal_tc_for_spd
            t = format_seasonal_tc_for_spd(sd["seasonal_tc_context"])
            if t:
                structured_sections.append(t)
        except Exception:
            pass

    # FEWS NET Most Likely projection for DR / PHASE3PLUS_IN_NEED
    if hazard == "DR" and metric == "PHASE3PLUS_IN_NEED":
        projection = _load_fewsnet_projection(iso3, forecast_keys)
        if projection:
            structured_sections.append(projection)

    # GDACS event history for FL/DR/TC
    if sd.get("gdacs_event_history"):
        try:
            forecast_cal_months: list[int] = []
            for fk in forecast_keys:
                _m = re.match(r"^\d{4}-(\d{2})", fk)
                if _m:
                    forecast_cal_months.append(int(_m.group(1)))
            if not forecast_cal_months:
                forecast_cal_months = list(range(1, 7))
            _gdacs_t = _format_gdacs_event_history_for_prompt(
                sd["gdacs_event_history"],
                forecast_cal_months,
            )
            if _gdacs_t:
                structured_sections.append(_gdacs_t)
        except Exception:
            pass

    structured_data_section = "\n\n".join(structured_sections) + "\n\n" if structured_sections else ""

    # PA resolution machine base rates + the process this forecast will be
    # scored by. Rendered immediately after the Resolver history rather than
    # among the structured injects, because it is the same kind of thing —
    # a prior anchor — and STEP 1 tells the model to build its prior from
    # what it finds there. Self-gating: "" for every hazard/metric the
    # machine does not resolve (see prompt_block.is_eligible), so questions
    # outside its scope assemble byte-identically to before.
    _haz_base_rates = _load_haz_base_rate_block(iso3, hazard, metric, forecast_keys)
    haz_base_rate_section = f"{_haz_base_rates}\n\n" if _haz_base_rates else ""

    # SPD self-driven web search instruction. Gated on the SAME check the
    # executor uses (forecaster.self_search.self_search_enabled) — the prompt
    # must never invite a NEED_WEB_EVIDENCE request the pipeline will refuse
    # (that drops the model's forecast with error `self_search_disabled`).
    spd_web_search_section = ""
    if _self_search_enabled():
        spd_web_search_section = (
            "\nWEB SEARCH: If you need to verify a specific recent data point, "
            "check a recent development, or resolve an ambiguity in the evidence, "
            "you may use web search. Limit to 1-2 targeted searches maximum. "
            "Do not search broadly or speculatively — only search when a specific "
            "factual question would materially change your probability distribution.\n\n"
        )

    if _self_search_enabled():
        need_evidence_block = (
            "If you need more evidence before forecasting, output EXACTLY one line:\n"
            "NEED_WEB_EVIDENCE: <your query>\n"
            "Otherwise, produce the forecast JSON.\n\n"
        )
    else:
        need_evidence_block = (
            "Web evidence requests are not available in this run. Do NOT output "
            "NEED_WEB_EVIDENCE. Produce the forecast JSON from the evidence provided.\n\n"
        )

    # Conditional research section — only render if research_json has PM,
    # seasonal, or tail pack data (full research brief no longer produced).
    research_section = ""
    _rj_keys = {"prediction_market_signals", "nmme_seasonal_outlook"}
    if any(research_json.get(k) for k in _rj_keys):
        research_section = (
            "Reference data:\n"
            "```json\n"
            f"{_json_dumps_for_prompt(research_json, indent=2)}\n"
            "```\n\n"
        )

    v3_order = _prompt_v3_order_enabled()
    # Positional referents differ by order: under V3 the method block precedes
    # the question data, so "above" flips to "below" (and vice versa). This is
    # the ONLY text difference between the two orders.
    history_ref = "below" if v3_order else "above"
    # base_rate_note travels with the base-rate data under V3 (data tail),
    # and stays inline in STEP 1 under legacy order.
    base_rate_note_in_data = (base_rate_note + "\n") if (v3_order and base_rate_note) else ""
    base_rate_note_in_method = "" if v3_order else base_rate_note
    # The per-question self-search escape (contains iso3/hazard) moves to the
    # dynamic tail under V3 so it can't split the static prefix.
    rc_self_search_in_method = "" if v3_order else rc_self_search_line
    rc_self_search_in_data = rc_self_search_line if v3_order else ""

    base_rate_text = _build_base_rate_text(history_summary, forecast_keys, iso3, hazard, metric)
    base_rate_text = _one_natural_hazard_anchor(
        base_rate_text, history_summary, bool(_haz_base_rates), iso3, hazard,
    )
    prior_anchor = load_prior_anchor(question)
    prior_anchor_section = ""
    if prior_anchor:
        # The trajectory block calls itself the prior anchor; with the
        # distribution below it is context, and one prior is enough.
        base_rate_text = base_rate_text.replace(
            "Use as your prior anchor.",
            "Use as context; your prior is the BASE-RATE DISTRIBUTION below.",
        )
        prior_anchor_section = render_prior_anchor_block(prior_anchor, forecast_keys) + "\n"

    # --- PROMPT_EXCERPT: spd_v2_start ---
    role_line = (
        "You are a careful probabilistic forecaster on a humanitarian early warning panel.\n\n"
    )
    task_line = (
        f"Your task is to produce a six-month PROBABILITY DISTRIBUTION over {n_buckets_lower} impact buckets for the question below, where each month’s probabilities sum to 1.0.\n\n"
    )
    question_data_block = (
        "Natural-language question:\n"
        f"\"{wording}\"\n\n"
        "Question metadata:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(question, indent=2)}\n"
        "```\n\n"
        f"{horizon_note}"
        f"{base_rate_text}\n\n"
        f"{prior_anchor_section}"
        f"{haz_base_rate_section}"
        f"{base_rate_note_in_data}"
        "HS triage output:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(hs_triage_entry, indent=2)}\n"
        "```\n\n"
        f"{research_section}"
        f"{rc_guidance}"
        f"{pm_signals_section}"
        f"{seasonal_outlook_section}"
        f"{structured_data_section}"
    )
    buckets_block = (
        "Buckets:\n"
        f"- These buckets represent {unit_phrase} at the country level.\n"
        f"- Bucket labels: {bucket_list_str}\n\n"
    )
    hazard_block = f"{hazard_reasoning_block}\n\n"
    method_and_output_block = (
        "FORECASTING METHOD: STRUCTURED BAYESIAN UPDATING\n\n"
        "You must follow these steps IN ORDER. Do not skip steps. Show your work for each step.\n\n"
        "STEP 1 — DECLARE YOUR PRIOR SPD\n"
        "Before considering ANY evidence, state your prior (base-rate) SPD for each month. "
        + (
            _PRIOR_ANCHOR_STEP1.format(ref=history_ref)
            if prior_anchor
            else (
                f"Derive this prior from the Resolver history summary {history_ref}. "
                "If Resolver history is available, convert the historical distribution into bucket probabilities. "
                "If history is missing or sparse, state an uninformative prior (e.g. heavy weight on the \"0\" "
                "bucket for countries with no recent events) and explain your reasoning.\n"
            )
        )
        +
        f"{base_rate_note_in_method}"
        "Write out the prior explicitly:\n"
        f"  Prior SPD: {prob_ph} for each month (or a single prior if months are similar).\n"
        "  Prior rationale: 1–2 sentences explaining why this is the right starting point.\n\n"
        "STEP 2 — IDENTIFY AND RANK UPDATE SIGNALS\n"
        "List the 3–6 most decision-relevant pieces of evidence from the structured data, HS triage, "
        "and any other context provided. For each signal, state:\n"
        "  - What the signal is (one line)\n"
        "  - Direction: does it push risk UP or DOWN relative to the prior?\n"
        "  - Magnitude: SMALL (shifts <5pp to any bucket), MODERATE (5–15pp), or LARGE (>15pp)\n"
        "  - Which months it affects (all, or specific months)\n"
        "Rank signals by magnitude (largest first). Ignore signals that do not meaningfully change the distribution.\n\n"
        "STEP 3 — SEQUENTIAL BAYESIAN UPDATE\n"
        "Starting from your prior, update the SPD one signal at a time, largest magnitude first. "
        "For each update:\n"
        "  a) State the signal being incorporated.\n"
        "  b) State which bucket(s) gain or lose probability mass, and roughly how much.\n"
        f"  c) Write the updated SPD after this signal: {prob_ph}.\n"
        "  d) Verify the updated SPD sums to ~1.0.\n"
        "You must show at least 2 explicit update steps (for your top 2 signals). "
        "Remaining signals can be incorporated in a single combined step if they are small.\n\n"
        + (_RC_SHIFT_METHOD_BLOCK if rc_shift_on else "")
        + "After all updates, state your POSTERIOR SPD for each month:\n"
        f"  Posterior SPD month_1: {prob_ph}\n"
        f"  Posterior SPD month_2: {prob_ph}\n"
        "  ... (all 6 months)\n\n"
        "STEP 4 — REFERENCE CLASS CHECK (outside view)\n"
        "Name 2–3 historical cases that are most similar to this country-hazard-period "
        "(e.g. \"Ethiopia drought PA, Oct–Mar 2021\" or \"Myanmar ACE fatalities, post-coup 2021\"). "
        "For each, briefly state:\n"
        "  - What happened (which bucket did the outcome land in?)\n"
        "  - How similar is the current situation? (high/medium/low similarity)\n"
        "Check: is your posterior SPD broadly consistent with these reference cases? "
        "If not, explain why the current situation justifies the deviation.\n\n"
        "STEP 5 — STRESS TEST AND RED TEAM\n"
        "A) Construct a specific scenario where the outcome lands in your LOWEST-probability "
        "bucket. How plausible is this scenario? Estimate its probability (e.g. 2%, 5%, 10%). "
        "Does your SPD allocate at least this much probability to that bucket?\n"
        "B) Argue the strongest case that your forecast is TOO LOW (i.e. you are underweighting "
        "higher buckets). In 1–2 sentences, what would have to be true?\n"
        "C) Argue the strongest case that your forecast is TOO HIGH (i.e. you are overweighting "
        "higher buckets). In 1–2 sentences, what would have to be true?\n"
        "D) Based on A–C, do you need to adjust your posterior? If so, state the adjustment "
        "and the revised SPD. If not, state \"No adjustment needed\" and why.\n\n"
        "STEP 5b — SCENARIO DECOMPOSITION CROSS-CHECK (optional but recommended for complex cases)\n"
        "Enumerate 3 concrete, mutually exclusive scenarios for the forecast period:\n"
        "  a) A LOW-impact scenario (outcome in the bottom two buckets). Describe in 1–2 sentences. "
        "Estimate its probability.\n"
        "  b) A MODERATE-impact scenario (outcome in a middle bucket). Describe. Estimate probability.\n"
        "  c) A HIGH-impact scenario (outcome in the top two buckets). Describe. Estimate probability.\n"
        "These scenario probabilities should roughly sum to ~100%. "
        "Cross-check: does your SPD match these scenario probabilities? "
        "If your high-impact scenario is 15% probable, the combined probability of the top two buckets should be roughly 15%. "
        "Resolve any mismatch.\n\n"
        "STEP 6 — POINT ESTIMATE CONSISTENCY CHECK\n"
        "For each month, state your single best-guess point estimate of the metric value "
        "(e.g. \"~15,000 people affected\" or \"~40 fatalities\"). State which bucket this falls in. "
        "Verify this is consistent with where your SPD places the most probability mass. "
        "If your point estimate falls in a middle bucket but your SPD puts 70% in the bottom bucket, "
        "something is wrong — resolve the inconsistency before proceeding.\n\n"
        f"{spd_web_search_section}"
        "STEP 7 — FINAL OUTPUT\n"
        "Only after completing Steps 1–6, produce the JSON output specified below. "
        "Your JSON probabilities must match the posterior SPD from Step 5 (with any adjustments "
        "from the stress test). Do not change the numbers at this stage.\n\n"
        f"{rc_self_search_in_method}"
        f"{need_evidence_block}"
        "Output instructions:\n"
        "- Return ONLY a single JSON object with this schema (no extra commentary):\n\n"
        + (
            "- `human_explanation` MUST include a sentence starting with \"RC:\" stating what HS RC flagged, whether you accepted it, and how it changed the SPD (shifted up/shifted down/two-sided/rebutted).\n"
            if rc_shift_on
            else "- `human_explanation` MUST include a sentence starting with \"RC:\" stating what HS RC flagged, whether you accepted it, and how it changed the SPD (widened/shifted/rebutted).\n"
        )
        + (
            "- `reasoning_trace.prior.spd` MUST match your Step 1 prior SPD (the base-rate-only distribution before any evidence updates).\n"
            "- `reasoning_trace.updates` MUST contain at least your top 2 update signals from Step 2, with numeric `delta` arrays showing how each signal shifted the distribution. Positive values in `delta` mean probability mass added to that bucket; negative means removed. Each `delta` array must sum to approximately 0. Write positive numbers plainly (`0.25`), never with a leading plus sign (`+0.25`) — a leading `+` is not valid JSON and makes the whole response unparseable.\n"
            "- `reasoning_trace.updates[].post_update_spd` is the running SPD after applying that signal — it must equal the previous SPD plus the delta (within rounding).\n"
            "- `reasoning_trace.point_estimate` and `point_estimate_bucket` must be consistent with your Step 6 check.\n"
            "- `reasoning_trace.rc_assessment` must state whether you accepted, rebutted, or partially accepted the HS regime change flag.\n"
            + (_RC_SHIFT_TRACE_INSTRUCTION if rc_shift_on else "")
            + "\n"
            "```json\n"
            "{\n"
            '  "reasoning_trace": {\n'
            '    "prior": {\n'
            f'      "spd": {prob_ph},\n'
            '      "rationale": "1-2 sentences explaining the prior derivation from base rate data"\n'
            "    },\n"
            '    "updates": [\n'
            "      {\n"
            '        "signal": "Short name of the evidence signal",\n'
            '        "direction": "UP or DOWN",\n'
            '        "magnitude": "SMALL or MODERATE or LARGE",\n'
            '        "months_affected": "all or 1-2 or specific month numbers",\n'
            f'        "delta": {delta_ph},\n'
            f'        "post_update_spd": {prob_ph}\n'
            "      }\n"
            "    ],\n"
            '    "point_estimate": "~NNN units (e.g. ~40 fatalities or ~15,000 people affected)",\n'
            '    "point_estimate_bucket": 3,\n'
            + (
                '    "rc_assessment": "accepted or rebutted or partially_accepted",\n'
                + _RC_SHIFT_SCHEMA_LINES
                if rc_shift_on
                else '    "rc_assessment": "accepted or rebutted or partially_accepted"\n'
            )
            + "  },\n"
            if track < 2
            else
            "- `reasoning_trace` must include `prior` (with `spd` and `rationale`) and `rc_assessment`. The `updates` array may be empty for Track 2 forecasts. `point_estimate` and `point_estimate_bucket` are optional.\n\n"
            "```json\n"
            "{\n"
            '  "reasoning_trace": {\n'
            '    "prior": {\n'
            f'      "spd": {prob_ph},\n'
            '      "rationale": "1-2 sentences"\n'
            "    },\n"
            '    "updates": [],\n'
            '    "rc_assessment": "accepted or rebutted or partially_accepted"\n'
            "  },\n"
        )
        + f'  "spds": {{\n'
        f'    "{example_key_1}": {{"buckets": [{bucket_list_str}], "probs": [{example_probs}]}},\n'
        f'    "{example_key_2}": {{"buckets": [{bucket_list_str}], "probs": [{example_probs}]}}\n'
        "  },\n"
        '  "human_explanation": "3-4 sentences summarising the base rate, key update signals, and why any large deviations from the base rate are justified."\n'
        "}\n"
        "```\n"
        f"Each `probs` array must contain exactly {n_buckets_lower} numbers between 0 and 1 that sum to ~1.0.\n"
        f"Each `delta` array must contain exactly {n_buckets_lower} numbers that sum to approximately 0.\n"
        "Do not include any text outside the JSON.\n"
    )
    # --- PROMPT_EXCERPT: spd_v2_end ---

    if not v3_order:
        # Legacy assembly — byte-identical to the pre-V3 prompt.
        prompt = (
            role_line
            + calibration_section
            + task_line
            + question_data_block
            + buckets_block
            + hazard_block
            + method_and_output_block
        )
        if return_parts:
            # No usefully cacheable prefix under legacy order.
            return "", prompt
        return prompt

    # V3 (static-first) assembly: everything stable within a
    # (hazard, metric, track) group leads; per-question data trails. The
    # calibration section is placed after the fully-static blocks so its
    # monthly refresh can't invalidate their cached span.
    prefix = (
        role_line
        + task_line
        + buckets_block
        + hazard_block
        + method_and_output_block
        + calibration_section
        + "The QUESTION DATA to forecast follows below.\n\n"
    )
    suffix = (
        question_data_block
        + rc_self_search_in_data
        + "END OF QUESTION DATA.\n"
        "Now apply the FORECASTING METHOD above (Steps 1–7) to the question data and "
        "produce ONLY the JSON object specified in the Output instructions.\n"
    )
    if return_parts:
        return prefix, suffix
    return prefix + suffix


def build_scenario_prompt(
    run_id: str,
    question: Dict[str, Any],
    ensemble_spd: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
) -> str:
    """Prompt the Scenario Writer to draft structured scenarios from ensemble outputs."""

    iso3 = question.get("iso3", "")
    hazard = (question.get("hazard_code") or "").upper()
    metric = (question.get("metric") or "").upper()
    wording = question.get("wording") or question.get("title") or ""
    scenario_text = hs_triage_entry.get("scenario_stub", "") if hs_triage_entry else ""
    rationale_text = question.get("forecaster_rationale") or ""

    # --- PROMPT_EXCERPT: scenario_start ---
    return (
        "You are a humanitarian analyst writing structured scenarios for senior decision-makers.\n\n"
        "Natural-language question:\n"
        f"\"{wording}\"\n\n"
        "Context for this question:\n"
        f"- Country: {iso3}\n"
        f"- Hazard: {hazard}\n"
        f"- Metric: {metric}\n\n"
        "Ensemble forecast (SPD) summary:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(ensemble_spd, indent=2)}\n"
        "```\n\n"
        "HS triage summary:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(hs_triage_entry, indent=2)}\n"
        "```\n\n"
        "Optional HS scenario stub:\n"
        f"\"\"\"{scenario_text}\"\"\"\n\n"
        "Forecaster rationale:\n"
        f"\"\"\"{rationale_text}\"\"\"\n\n"
        "Your task is to produce **one scenario** (primary) for the forecast period.\n"
        "The scenario must be returned as a single JSON object with the following schema (no extra text):\n\n"
        "```json\n"
        "{\n"
        '  "primary": {\n'
        '    "bucket_label": "bucket_3",\n'
        '    "probability": 0.6,\n'
        '    "context": ["• brief bullet about context", "• another bullet"],\n'
        '    "needs": {\n'
        '      "WASH": ["• bullet", "• bullet"],\n'
        '      "Health": ["• bullet"],\n'
        '      "Nutrition": ["• bullet"],\n'
        '      "Protection": ["• bullet"],\n'
        '      "Education": ["• bullet"],\n'
        '      "Shelter": ["• bullet"],\n'
        '      "FoodSecurity": ["• bullet"]\n'
        '    },\n'
        '    "operational_impacts": ["• bullet about ops impact", "• another bullet"]\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "Guidance:\n"
        "- **Context**: bullets describing how the situation looks over the forecast period (conflict, climate, displacement, etc.).\n"
        "- **Humanitarian Needs**: bullets for each sector (WASH, Health, Nutrition, Protection, Education, Shelter, Food Security), focusing on the main needs implied by the forecast.\n"
        "- **Operational Impacts**: bullets on what this means for access, surge, supply chains, partnerships, and programmatic choices.\n"
        "- Ensure bullets are concise and specific (no long paragraphs).\n"
        "- Align the scenario with the ensemble SPD (bucket_label and probability) and HS triage evidence.\n"
        + (
            "\nIf you need more evidence before drafting the scenarios, output EXACTLY one line:\n"
            "NEED_WEB_EVIDENCE: <your query>\n"
            "Otherwise, return only the JSON object above.\n"
            if _self_search_enabled()
            else
            "\nWeb evidence requests are not available in this run. Do NOT output "
            "NEED_WEB_EVIDENCE. Return only the JSON object above.\n"
        )
    )
    # --- PROMPT_EXCERPT: scenario_end ---
