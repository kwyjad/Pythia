# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A starting pack of the pipeline's structured data (Oct 2026, review Part 6).

Sibyl's trials work from the open web. The pipeline already holds, for every
country, a dozen structured feeds the ensemble's prompt carries: food
security, INFORM severity, GDACS history, HDX Signals, ENSO, the NMME
outlook, CrisisWatch, GDELT, ACLED political events, ACAPS. Whether giving
those to Sibyl as a starting pack helps its forecasts, or only pulls them
toward the ensemble's, is a question to measure rather than assume. So a
hashed half of questions (``SIBYL_PACK_SHARE``, salt ``sibyl_pack:``) gets
the pack, and the other half does not.

* The pack is built on the main thread from
  ``forecaster/cli._load_structured_data`` (called with no HS run and no RC
  level, so the ensemble's own grounding packs and adversarial checks are
  never loaded) plus the NMME outlook loader, and rendered with the same
  formatters the SPD prompt uses, under the same hazard gates.
* Only ``PACK_SECTIONS`` may appear: never the HS grounding, the adversarial
  check, RC or triage output, the ensemble's base rate, history, advice or
  forecasts, or the ReliefWeb reports (Sibyl searches ReliefWeb itself).
  Conflict forecasts (VIEWS, conflictforecast.org, ACLED CAST) only with
  ``SIBYL_PACK_INCLUDE_FORECASTS=1``. Food security is left out of a drought
  question when the resolver reading already shows its Phase 3+ rows.
* The block sits in the question segment after the resolver reading, capped
  at ``SIBYL_PACK_MAX_CHARS`` (24,000) by dropping whole sections from the
  end of the priority order, each dropped section named.
* Nothing in backtest: the tables hold data published after the as-of date.

``pack_comparison`` reports the arms apart for selected questions and for
controls: immediate measures from 10 questions an arm, and the gain over
Sibyl's own reference from 20 scored questions an arm, otherwise "not yet".
Nothing switches the pack on for every question by itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sibyl import config as _cfg
from sibyl.leakage import is_backtest

logger = logging.getLogger(__name__)

ARM_PACK = "pack"
ARM_NO_PACK = "no_pack"
ARM_PACK_EMPTY = "pack_empty"
PACK_SALT = "sibyl_pack:"

HEADING = "=== STRUCTURED DATA HELD BY THE PIPELINE (as of {as_of}) ==="
PREFACE = (
    "Feeds the forecasting pipeline already holds for this country, with their own "
    "dates. They are a starting point for your research, not evidence you have read: "
    "check what matters, date what you use, and record in the ledger only what you "
    "verify. They count toward neither the search nor the document requirement."
)

# Priority order, highest first: when the cap binds, sections leave from the
# end. Each entry: key in the structured-data dict, hazards it applies to
# (None = every hazard), and a short label for the dropped-sections line.
PACK_SECTIONS: Tuple[Tuple[str, Optional[Tuple[str, ...]], str], ...] = (
    ("crisiswatch", ("ACE",), "ICG CrisisWatch"),
    ("fewsnet_food_security", None, "food security"),
    ("inform_severity", None, "INFORM severity"),
    ("gdacs_event_history", ("FL", "DR", "TC"), "GDACS event history"),
    ("hdx_signals", None, "HDX Signals"),
    ("nmme", ("FL", "DR", "TC", "HW"), "NMME seasonal outlook"),
    ("enso_context", ("TC", "FL", "DR", "HW"), "ENSO"),
    ("seasonal_tc_context", ("TC",), "seasonal cyclone outlooks"),
    ("acled_political_events", ("ACE", "DI"), "ACLED political events"),
    ("gdelt_conflict_indicators", ("ACE",), "GDELT conflict indicators"),
    ("acaps_risk_radar", None, "ACAPS Risk Radar"),
    ("acaps_monitoring", None, "ACAPS daily monitoring"),
    ("conflict_forecasts", ("ACE", "DI"), "conflict forecasts"),
)
FORECAST_SECTIONS = frozenset({"conflict_forecasts"})
# Never in the pack, whatever a loader returns: the ensemble's own research.
NEVER_SECTIONS = frozenset({
    "hazard_grounding", "adversarial_check", "reliefweb_reports", "ipc_phases",
})


@dataclass
class Pack:
    arm: Optional[str] = None
    text: str = ""
    sections: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    chars: int = 0
    error: Optional[str] = None

    def record(self) -> Dict[str, Any]:
        return {"sections": self.sections, "dropped": self.dropped,
                "chars": self.chars, "error": self.error}


def pack_arm(question_id: Optional[str], share: Optional[float] = None) -> str:
    """``"pack"`` or ``"no_pack"``; its own salt, so it is independent of the
    advice arm and of the standard track's arms. Controls are hashed too."""
    share = float(_cfg.PACK_SHARE if share is None else share)
    if not question_id or share <= 0:
        return ARM_NO_PACK
    if share >= 1:
        return ARM_PACK
    frac = int(hashlib.sha1(f"{PACK_SALT}{question_id}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return ARM_PACK if frac < share else ARM_NO_PACK


def _render_nmme(data: Dict[str, Any]) -> str:
    from forecaster.prompts import NMME_UNITS_NOTE  # noqa: PLC0415

    lines = ["SEASONAL CLIMATE OUTLOOK (NMME multi-model ensemble mean):"]
    for k, v in data.items():
        lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
    lines.append(NMME_UNITS_NOTE)
    return "\n".join(lines)


def _render_gdacs(data: Any, forecast_keys: Sequence[str]) -> str:
    from forecaster.prompts import _format_gdacs_event_history_for_prompt  # noqa: PLC0415

    months = [int(k[5:7]) for k in forecast_keys if len(k) >= 7 and k[5:7].isdigit()]
    return _format_gdacs_event_history_for_prompt(data, months or list(range(1, 7)))


def render_section(key: str, value: Any, forecast_keys: Sequence[str]) -> str:
    """One section in the SPD prompt's own rendering. '' when it renders nothing."""
    if not value:
        return ""
    if key == "fewsnet_food_security":
        from pythia.food_security import format_food_security_for_spd as f  # noqa: PLC0415
    elif key == "inform_severity":
        from pythia.acaps import format_inform_severity_for_spd as f  # noqa: PLC0415
    elif key == "acaps_risk_radar":
        from pythia.acaps import format_risk_radar_for_spd as f  # noqa: PLC0415
    elif key == "acaps_monitoring":
        from pythia.acaps import format_daily_monitoring_for_spd as f  # noqa: PLC0415
    elif key == "acled_political_events":
        from pythia.acled_political import format_political_events_for_spd as f  # noqa: PLC0415
    elif key == "seasonal_tc_context":
        from horizon_scanner.seasonal_tc import format_seasonal_tc_for_spd as f  # noqa: PLC0415
    elif key == "gdacs_event_history":
        return _render_gdacs(value, forecast_keys) or ""
    elif key == "nmme":
        return _render_nmme(value) if isinstance(value, dict) else ""
    else:
        # Pre-formatted by its loader (crisiswatch, hdx, enso, gdelt, conflict).
        return str(value)
    return f(value) or ""


def _default_loader(iso3: str, hazard: str) -> Dict[str, Any]:
    from forecaster.cli import _load_structured_data  # noqa: PLC0415

    sd = dict(_load_structured_data(iso3, hazard, hs_run_id=None, rc_level=None) or {})
    try:
        from horizon_scanner.seasonal_context import CLIMATE_HAZARDS, load_seasonal_forecasts  # noqa: PLC0415

        if hazard in CLIMATE_HAZARDS:
            nmme = load_seasonal_forecasts(iso3)
            if nmme:
                sd["nmme"] = nmme
    except Exception as exc:  # noqa: BLE001
        logger.debug("sibyl.pack: NMME load failed for %s: %s", iso3, exc)
    return sd


def build_pack(
    question: Any,
    as_of: date,
    *,
    forecast_keys: Sequence[str] = (),
    resolver_reading_shown: bool = False,
    loader: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    max_chars: Optional[int] = None,
) -> Pack:
    """The pack for one question. Never raises.

    The arm is decided first; a ``no_pack`` question loads nothing. A pack
    question whose sections all came back empty is ``pack_empty``. Backtest
    renders nothing and records no arm.
    """
    if _cfg.BACKTEST_MODE or is_backtest(as_of):
        return Pack()
    arm = pack_arm(getattr(question, "question_id", None))
    if arm != ARM_PACK:
        return Pack(arm=arm)
    hazard = str(getattr(question, "hazard_code", "") or "").upper()
    metric = str(getattr(question, "metric", "") or "").upper()
    iso3 = str(getattr(question, "iso3", "") or "").upper()
    cap = int(_cfg.PACK_MAX_CHARS if max_chars is None else max_chars)
    try:
        sd = (loader or _default_loader)(iso3, hazard) or {}
    except Exception as exc:  # noqa: BLE001
        from sibyl.resolver_reading import _scrub  # noqa: PLC0415

        logger.warning("sibyl.pack: load failed for %s: %s", iso3, exc)
        return Pack(arm=ARM_PACK_EMPTY, error=_scrub(exc))
    rendered: List[Tuple[str, str, str]] = []
    for key, hazards, label in PACK_SECTIONS:
        if key in NEVER_SECTIONS:
            continue
        if key in FORECAST_SECTIONS and not _cfg.PACK_INCLUDE_FORECASTS:
            continue
        if hazards is not None and hazard not in hazards:
            continue
        if (key == "fewsnet_food_security" and resolver_reading_shown
                and hazard == "DR" and metric == "PHASE3PLUS_IN_NEED"):
            continue  # the resolver reading already shows its Phase 3+ rows
        try:
            text = render_section(key, sd.get(key), forecast_keys).strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("sibyl.pack: %s failed to render: %s", key, exc)
            text = ""
        if text:
            rendered.append((key, label, text))
    if not rendered:
        return Pack(arm=ARM_PACK_EMPTY)
    head = HEADING.format(as_of=as_of.isoformat()) + "\n" + PREFACE
    kept = list(rendered)
    dropped: List[str] = []

    def _assemble(items: List[Tuple[str, str, str]], drops: List[str]) -> str:
        body = "\n\n".join(t for _k, _l, t in items)
        tail = (f"\n\n(Left out for length: {', '.join(drops)}.)" if drops else "")
        return "\n\n" + head + "\n\n" + body + tail

    text = _assemble(kept, dropped)
    while len(text) > cap and len(kept) > 1:
        _k, label, _t = kept.pop()
        dropped.insert(0, label)
        text = _assemble(kept, dropped)
    if len(text) > cap:
        # One section alone over the cap: nothing fits, and nothing is cut short.
        dropped.insert(0, kept[0][1])
        return Pack(arm=ARM_PACK_EMPTY, dropped=dropped)
    return Pack(arm=ARM_PACK, text=text, sections=[k for k, _l, _t in kept],
                dropped=dropped, chars=len(text))


# --- comparison --------------------------------------------------------------------

def _mean_interval(values: Sequence[float], seed: int) -> Tuple[float, float, float]:
    from sibyl.advice import BOOTSTRAP_DRAWS, INTERVAL  # noqa: PLC0415

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_DRAWS, len(arr)))].mean(axis=1)
    lo, hi = np.quantile(boot, INTERVAL)
    return float(arr.mean()), float(lo), float(hi)


def _diff_interval(a: Sequence[float], b: Sequence[float], seed: int) -> Tuple[float, float, float]:
    """Mean of *a* minus mean of *b*; each arm's questions resampled apart."""
    from sibyl.advice import BOOTSTRAP_DRAWS, INTERVAL  # noqa: PLC0415

    xa, xb = np.array(a, dtype=float), np.array(b, dtype=float)
    rng = np.random.default_rng(seed)
    ba = xa[rng.integers(0, len(xa), size=(BOOTSTRAP_DRAWS, len(xa)))].mean(axis=1)
    bb = xb[rng.integers(0, len(xb), size=(BOOTSTRAP_DRAWS, len(xb)))].mean(axis=1)
    lo, hi = np.quantile(ba - bb, INTERVAL)
    return float(xa.mean() - xb.mean()), float(lo), float(hi)


def _jsd(p: Sequence[float], q: Sequence[float]) -> Optional[float]:
    try:
        from sibyl.spd import _js_divergence  # noqa: PLC0415

        if p and q and len(p) == len(q):
            return float(_js_divergence(list(p), list(q)))
    except Exception:  # noqa: BLE001
        return None
    return None


def _immediate(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    trials = row.get("trials") or []
    n = len(trials)
    ref = ((row.get("reference") or {}).get("by_month") or {}).get("1")
    raw = ((row.get("raw") or {}).get("vectors") or {}).get("1")
    return {
        "jsd_vs_standard": row.get("jsd_vs_standard"),
        "jsd_inter_trial": row.get("jsd_inter_trial"),
        "jsd_month1_from_reference": _jsd(raw, ref) if raw and ref else None,
        "docs_per_trial": (sum(int(t.get("n_docs_read") or 0) for t in trials) / n) if n else None,
        "steps_per_trial": (sum(int(t.get("steps_used") or 0) for t in trials) / n) if n else None,
    }


IMMEDIATE_MEASURES = ("jsd_vs_standard", "jsd_inter_trial", "jsd_month1_from_reference",
                      "docs_per_trial", "steps_per_trial")


def pack_comparison(con: Any, *, include_test: bool = False,
                    min_immediate: Optional[int] = None,
                    min_scored: Optional[int] = None) -> Dict[str, Any]:
    """The pack arms compared, selected questions and controls apart. Never raises.

    Immediate measures (JSD vs the standard track, inter-trial JSD, month-1
    JSD of the raw pool from the reference, documents and steps per trial)
    are shown from *min_immediate* (10) questions an arm. The outcome
    measure is the gain over Sibyl's own reference, the per-question mean of
    ``sibyl`` minus ``__ext_sibyl_ref`` Brier over resolved horizons, shown
    from *min_scored* (20) scored questions an arm, with the pack-minus-no-pack
    difference and a 90% interval resampling each arm's questions. "Not yet"
    below either threshold, with no number.
    """
    min_i = int(_cfg.PACK_MIN_QUESTIONS_IMMEDIATE if min_immediate is None else min_immediate)
    min_s = int(_cfg.PACK_MIN_QUESTIONS_SCORED if min_scored is None else min_scored)
    out: Dict[str, Any] = {"min_questions_immediate": min_i, "min_questions_scored": min_s,
                           "groups": {}}
    try:
        cols = {str(r[1]).lower() for r in con.execute(
            "PRAGMA table_info('sibyl_forecasts')").fetchall()}
        if "pack_arm" not in cols:
            return out
        test_f = " AND NOT COALESCE(is_test, FALSE)" if not include_test else ""
        ev = " AND COALESCE(evidence_ok, TRUE)" if "evidence_ok" in cols else ""
        rows = con.execute(
            f"""
            SELECT question_id, pack_arm, selection_pass, js_divergence_vs_standard,
                   js_divergence_inter_trial, trials_json, raw_by_month_json, reference_json
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY question_id
                    ORDER BY created_at DESC NULLS LAST, sibyl_run_id DESC) AS rn
                FROM sibyl_forecasts
                WHERE status = 'ok' AND pack_arm IS NOT NULL{ev}{test_f}
            ) WHERE rn = 1
            """
        ).fetchall()
        has_scores = bool(con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'scores'").fetchone())
        gains: Dict[str, float] = {}
        if has_scores:
            from pythia.tools.scoring_class import scored_only_clause  # noqa: PLC0415

            s_test = "" if include_test else " AND NOT COALESCE(a.is_test, FALSE)"
            s_test += scored_only_clause(con, "a")
            for qid, v in con.execute(
                f"""
                SELECT a.question_id, AVG(a.value - b.value)
                FROM scores a JOIN scores b
                  ON a.question_id = b.question_id AND a.horizon_m = b.horizon_m
                 AND a.score_type = b.score_type
                WHERE a.model_name = 'sibyl' AND b.model_name = '__ext_sibyl_ref'
                  AND b.run_id IS NULL AND a.score_type = 'brier'{s_test}
                GROUP BY 1
                """
            ).fetchall():
                if v is not None and math.isfinite(float(v)):
                    gains[str(qid)] = float(v)
        by_group: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for qid, arm, sel, jsd_std, jsd_it, tj, rj, refj in rows:
            group = "control" if sel == "control" else "selected"
            rec = {"question_id": qid, "jsd_vs_standard": jsd_std, "jsd_inter_trial": jsd_it,
                   "trials": json.loads(tj) if tj else [], "raw": json.loads(rj) if rj else {},
                   "reference": json.loads(refj) if refj else {}}
            by_group.setdefault(group, {}).setdefault(arm, []).append(rec)
        for g_i, (group, arms) in enumerate(sorted(by_group.items())):
            g_out: Dict[str, Any] = {"arms": {}}
            for a_i, (arm, recs) in enumerate(sorted(arms.items())):
                # Question order fixed, so the seeded interval is reproducible.
                recs = sorted(recs, key=lambda r: str(r["question_id"]))
                vals = [_immediate(r) for r in recs]
                a_out: Dict[str, Any] = {"n_questions": len(recs)}
                if len(recs) < min_i:
                    a_out["immediate"] = {"status": "not_yet"}
                else:
                    a_out["immediate"] = {"status": "ok"}
                    for m in IMMEDIATE_MEASURES:
                        xs = [v[m] for v in vals if v[m] is not None]
                        a_out["immediate"][m] = (float(np.mean(xs)) if xs else None)
                g_gains = [gains[r["question_id"]] for r in recs if r["question_id"] in gains]
                a_out["n_scored"] = len(g_gains)
                if len(g_gains) < min_s:
                    a_out["gain_over_reference"] = {"status": "not_yet"}
                else:
                    m, lo, hi = _mean_interval(g_gains, seed=20261009 + 10 * g_i + a_i)
                    a_out["gain_over_reference"] = {"status": "ok", "brier_diff": m,
                                                    "lo": lo, "hi": hi}
                a_out["_gains"] = g_gains
                g_out["arms"][arm] = a_out
            p = g_out["arms"].get(ARM_PACK, {}).get("_gains") or []
            n = g_out["arms"].get(ARM_NO_PACK, {}).get("_gains") or []
            if len(p) >= min_s and len(n) >= min_s:
                d, lo, hi = _diff_interval(p, n, seed=20261019 + g_i)
                g_out["pack_minus_no_pack"] = {"status": "ok", "brier_diff": d, "lo": lo, "hi": hi}
            else:
                g_out["pack_minus_no_pack"] = {"status": "not_yet", "n_pack": len(p),
                                               "n_no_pack": len(n)}
            for a_out in g_out["arms"].values():
                a_out.pop("_gains", None)
            out["groups"][group] = g_out
    except Exception as exc:  # noqa: BLE001
        logger.warning("sibyl.pack: comparison failed: %s", exc)
        out["error"] = str(exc)[:200]
    return out
