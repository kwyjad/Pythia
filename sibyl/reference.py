# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl's reference: the mechanical forecast it starts from and is pooled with.

Until October 2026 Sibyl's outside view was a short text block (for conflict,
three numbers) and the agent's distribution was published as it stood. A
backtest on ACLED history (8,371 country-forecasts, Mar 2021 - Dec 2025,
production timing) gave Brier 0.390 for the bucket shares of a country's last
12 months, 0.384 for a 75/25 pool of that with the level-transition
reference, and 0.470 for level-and-volatility. Three to eight months after
the last observed month the fatality bucket is unchanged 67% of the time.

So Sibyl now has its OWN prior (owner decision; the ensemble's anchor is
untouched). ``build_reference`` returns one bucket vector per window month:

* ACE/FATALITIES: ``base_rate_spd.reference_pool_spds`` (0.75 x the 12-month
  shares + 0.25 x level_transition).
* FL/PA, TC/PA: the per-calendar-month vectors of ``_seasonal_pa``
  (that month's event rate and PA records; pooled severity below three).
* DR/PHASE3PLUS_IN_NEED: ``SIBYL_DR_PERSISTENCE_WEIGHT`` (0.5) x the
  persistence vector of the last observed value + the rest x the 36-month
  Phase 3+ history vector, the same for all six months unless
  ``SIBYL_DR_PERSISTENCE_WEIGHTS`` gives one weight per month. The weight is
  a starting value; sibyl/reference_backtest.py scores the alternatives.

It also renders the block the prompt shows in place of the old outside view.
Every sentence describing the distribution is read off the vectors printed
beside it. Returns None when no history exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

MONTHS = (1, 2, 3, 4, 5, 6)


@dataclass
class Reference:
    """The reference for one question."""

    by_month: Dict[int, List[float]]
    source: str
    detail: Dict[str, Any]
    history: List[Tuple[str, Optional[float]]] = field(default_factory=list)
    current_value: Optional[float] = None
    prompt_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "by_month": {str(m): v for m, v in sorted(self.by_month.items())},
            "history": [[ym, v] for ym, v in self.history],
            "current_value": self.current_value,
            "detail": self.detail,
        }


def _bucket(value: float, metric: str) -> Optional[int]:
    from pythia.buckets import interior_thresholds_for  # noqa: PLC0415

    if value is None:
        return None
    j = 0
    for t in interior_thresholds_for(metric):
        if float(value) >= t:
            j += 1
    return j


def _pa_history(con, iso3: str, hazard: str, months: Sequence[str]) -> List[Tuple[str, Optional[float]]]:
    from pythia.tools.base_rate_spd import _pa_metric_in_clause, _table_exists  # noqa: PLC0415

    if not months or not _table_exists(con, "facts_resolved"):
        return [(m, None) for m in months]
    rows = con.execute(
        f"""
        SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS m, MAX(value)
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = ? AND {_pa_metric_in_clause()}
          AND substr(CAST(ym AS VARCHAR), 1, 7) >= ? AND substr(CAST(ym AS VARCHAR), 1, 7) <= ?
        GROUP BY m
        """,
        [iso3, hazard, months[0], months[-1]],
    ).fetchall()
    got = {str(m): float(v) for m, v in rows if v is not None}
    return [(m, got.get(m)) for m in months]


def _phase3_series(con, iso3: str, before_ym: str) -> List[Tuple[str, Optional[float]]]:
    from pythia.tools.base_rate_spd import _table_exists  # noqa: PLC0415

    if not _table_exists(con, "facts_resolved"):
        return []
    rows = con.execute(
        """
        SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS m, MAX(value)
        FROM facts_resolved
        WHERE iso3 = ? AND hazard_code = 'DR' AND lower(metric) = 'phase3plus_in_need'
          AND value IS NOT NULL AND substr(CAST(ym AS VARCHAR), 1, 7) < ?
        GROUP BY m ORDER BY m DESC LIMIT 12
        """,
        [iso3, before_ym],
    ).fetchall()
    return [(str(m), float(v)) for m, v in reversed(rows)]


def _norm(vec: Sequence[float]) -> List[float]:
    z = sum(float(x) for x in vec)
    return [float(x) / z for x in vec] if z > 0 else list(vec)


def build_reference(
    con,
    question: Any,
    forecast_keys: Sequence[str],
    as_of: date,
    *,
    known_at: Any = None,
) -> Optional[Reference]:
    """The reference for months 1-6, or None when no history exists.

    ``forecast_keys`` are the window's six 'YYYY-MM' months (month 1 first).
    ``known_at`` is when the forecast is made (the as-of date); the ACLED
    level month must have settled by then.
    """
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415

    hz = str(question.hazard_code or "").upper()
    metric = str(question.metric or "").upper()
    iso3 = str(question.iso3 or "").upper()
    if not forecast_keys:
        return None
    window = str(forecast_keys[0])[:7]
    known = known_at or as_of
    try:
        if hz == "ACE" and metric == "FATALITIES":
            spds, source, detail = brs.reference_pool_spds(con, iso3, window, MONTHS, known_at=known)
            if not spds:
                return None
            c = detail.get("conflictology") or {}
            history = list(zip(c.get("months") or [], c.get("values") or []))
            current = c.get("level_value")
            return _finish(Reference(by_month=spds, source=source, detail=detail,
                                     history=history, current_value=current), question, forecast_keys)

        if metric == "PA" and hz in ("FL", "TC"):
            probs, source, detail = brs.base_rate_spd(con, iso3, hz, metric, window)
            if not probs:
                return None
            pbm = detail.get("probs_by_month") or {}
            by_month = {
                i + 1: list(pbm.get(str(ym)[:7]) or probs) for i, ym in enumerate(forecast_keys[:6])
            }
            hist_months = brs._window_months(window, 12)
            history = _pa_history(con, iso3, hz, hist_months)
            return _finish(Reference(by_month=by_month, source=source, detail=detail,
                                     history=history, current_value=None), question, forecast_keys)

        if hz == "DR" and metric == "PHASE3PLUS_IN_NEED":
            from pythia.tools.score_baselines import persistence_spd  # noqa: PLC0415

            hist_probs, hsrc, hdetail = brs._phase3_history(con, iso3, window)
            last = brs.last_observed_value(con, iso3, hz, metric, window)
            pers = persistence_spd(last[0], metric) if last else None
            if not hist_probs and not pers:
                return None
            # One weight per window month (SIBYL_DR_PERSISTENCE_WEIGHTS); at
            # the default all six equal SIBYL_DR_PERSISTENCE_WEIGHT and the
            # vectors, source and detail are as they were.
            weights = tuple(float(x) for x in _cfg.DR_PERSISTENCE_WEIGHTS)
            w = weights[0]
            same = all(x == w for x in weights)
            by_month: Dict[int, List[float]] = {}
            if hist_probs and pers:
                for m in MONTHS:
                    wm = weights[m - 1]
                    by_month[m] = _norm([wm * a + (1.0 - wm) * b for a, b in zip(pers, hist_probs)])
                source = (f"pool:persistence_{w:g}+{hsrc}" if same
                          else f"pool:persistence_schedule+{hsrc}")
            else:
                vec = _norm(pers or hist_probs)
                by_month = {m: list(vec) for m in MONTHS}
                source = f"persistence:{last[1]}" if pers else hsrc
            detail = {
                "method": "persistence_x_phase3_history",
                "persistence_weight": w if (hist_probs and pers) else (1.0 if pers else 0.0),
                "last_observed": list(last) if last else None,
                "history": hdetail,
            }
            if hist_probs and pers and not same:
                detail["persistence_weights"] = list(weights)
            history = _phase3_series(con, iso3, window)
            return _finish(Reference(by_month={m: list(v) for m, v in by_month.items()}, source=source,
                                     detail=detail, history=history,
                                     current_value=(last[0] if last else None)),
                           question, forecast_keys)
    except Exception as exc:  # noqa: BLE001 - a reference failure must not stop a question
        logger.warning("sibyl.reference: %s failed: %s", getattr(question, "question_id", "?"), exc)
        return None
    return None


def _pct(x: float) -> str:
    v = 100.0 * float(x)
    if 0 < v < 1:
        return "<1%"
    return f"{v:.0f}%"


def _fmt_value(v: Optional[float]) -> str:
    if v is None:
        return "no record"
    return f"{v:,.0f}"


def _shares(vec: Sequence[float], current_bucket: int) -> Tuple[float, float, float]:
    stay = float(vec[current_bucket])
    rise = float(sum(vec[current_bucket + 1:]))
    fall = float(sum(vec[:current_bucket]))
    return stay, rise, fall


def render_reference_block(ref: Reference, question: Any, forecast_keys: Sequence[str]) -> str:
    """The prompt block: history, the reference vectors, and what they say."""
    from pythia.buckets import labels_for  # noqa: PLC0415

    metric = str(question.metric or "").upper()
    hz = str(question.hazard_code or "").upper()
    labels = labels_for(metric)
    m1, m6 = str(forecast_keys[0])[:7], str(forecast_keys[min(5, len(forecast_keys) - 1)])[:7]
    lines: List[str] = []
    if hz == "ACE":
        lines.append(
            "Source: ACLED monthly fatalities, all event types. Reference = 0.75 x the "
            "bucket shares of the last 12 months + 0.25 x the bucket moves of past months "
            "that started in the current bucket."
        )
    elif hz == "DR":
        lines.append(
            "Source: FEWS NET / IPC Phase 3+ current-situation figures. Reference = an even "
            "pool of 'the last figure persists' and the bucket shares of the last 36 months "
            "of figures."
        )
    else:
        lines.append(
            "Source: people-affected records (IFRC GO, then IDMC) and the GDACS event rate "
            "for each calendar month. Reference = P(no impact or no record) from that "
            "month's event rate; the rest spread like that month's past records."
        )
    if ref.history:
        lines.append("")
        lines.append("Last monthly values, oldest first (bucket in brackets):")
        for ym, v in ref.history:
            b = _bucket(v, metric) if v is not None else None
            lab = f" [{labels[b]}]" if b is not None and b < len(labels) else ""
            lines.append(f"  {ym}: {_fmt_value(v)}{lab}")
        if hz == "ACE":
            lines.append(
                "  Note: ACLED keeps adding events to a month for weeks after it ends; the "
                "newest one or two months above may still rise a little."
            )
    v1 = ref.by_month.get(1) or []
    v6 = ref.by_month.get(6) or v1
    lines.append("")
    lines.append(f"Reference probability by bucket (month 1 = {m1}, month 6 = {m6}):")
    lines.append("  bucket | month 1 | month 6")
    for i, lab in enumerate(labels):
        a = _pct(v1[i]) if i < len(v1) else "-"
        b = _pct(v6[i]) if i < len(v6) else "-"
        lines.append(f"  {lab} | {a} | {b}")
    cb = _bucket(ref.current_value, metric) if ref.current_value is not None else None
    if cb is not None and v1:
        s1 = _shares(v1, cb)
        s6 = _shares(v6, cb)
        lines.append("")
        lines.append(
            f"Read off these vectors, against the current bucket ({labels[cb]}): "
            f"month 1 stays {_pct(s1[0])}, rises {_pct(s1[1])}, falls {_pct(s1[2])}; "
            f"month 6 stays {_pct(s6[0])}, rises {_pct(s6[1])}, falls {_pct(s6[2])}."
        )
    elif v1:
        lines.append("")
        lines.append(
            f"Read off these vectors: no impact or no record {_pct(v1[0])} in month 1 "
            f"and {_pct(v6[0])} in month 6."
        )
    return "\n".join(lines)


def _finish(ref: Reference, question: Any, forecast_keys: Sequence[str]) -> Reference:
    ref.by_month = {m: _norm(v) for m, v in ref.by_month.items()}
    ref.prompt_text = render_reference_block(ref, question, forecast_keys)
    return ref


NO_REFERENCE_TEXT = (
    "No mechanical reference exists for this question: the database holds no usable "
    "history for it. Build your distribution from research alone, keep it wide, and "
    "say in baserate_reconciliation what you anchored on."
)
