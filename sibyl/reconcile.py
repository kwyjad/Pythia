# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The reconciler trial's brief (Oct 2026, review Part 4).

When the production trials of a question disagree (the extra-trial rule
``disagreement``), the extra trials are lane R and lane E rather than D and E.
Lane R is one more trial that starts from the disagreement: it is shown what
the earlier trials forecast and what they recorded, names the factual points
they differ on, and searches for evidence on those points before it fills
its plan. Bridgewater's AIA Forecaster gained from a supervisor that ran new
searches on the point its forecasts disagreed about; one that only re-read
the forecasts did worse than the plain mean. So R is a trial, pooled with
equal weight, never a judge over the pool.

``build_brief`` runs on the main thread from the valid production trials and
is pure: no DB, no network. It renders

* per trial: lane, month-1 and month-6 ``p_zero``, median, 0.05 and 0.95
  quantiles, its ``baserate_reconciliation`` and its plan findings;
* the merged ledger, de-duplicated by URL and quote, each item tagged with
  the lanes that recorded it, inside ``SIBYL_RECONCILE_BRIEF_MAX_CHARS``
  (12,000): tier 1 and 2 items and dated figures first, whole items dropped
  and counted, never one cut short;
* the dispute in numbers, stated by the code: the medians by lane for each
  month, the largest pairwise month-1 JSD, and the buckets that carry the
  difference.

Items in the brief are not documents read and not searches: an R trial that
makes no successful search of its own fails the evidence gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

RECONCILE_LANE = "R"
_DIGIT = re.compile(r"\d")


@dataclass
class ReconcileBrief:
    text: str
    dispute: Dict[str, Any] = field(default_factory=dict)
    n_items: int = 0
    n_items_dropped: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)


def _q(qpos: Dict[Any, Any], level: float) -> Optional[float]:
    for k, v in (qpos or {}).items():
        try:
            if abs(float(k) - level) < 1e-9:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:.2g}"


def _trial_lines(trial: Any) -> List[str]:
    from sibyl.trials import month_median  # noqa: PLC0415

    lines = [f"Trial {trial.trial_index}, lane {trial.lane or '?'}:"]
    for m in (1, 6):
        mb = (trial.month_beliefs or {}).get(m)
        if mb is None:
            continue
        med = month_median(mb.p_zero, mb.quantiles_positive)
        lines.append(
            f"  month {m}: p_zero {mb.p_zero:.2f}, median {_fmt(med)}, positive 0.05 "
            f"{_fmt(_q(mb.quantiles_positive, 0.05))}, 0.95 {_fmt(_q(mb.quantiles_positive, 0.95))}"
        )
    last = trial.belief_trace[-1].belief if trial.belief_trace else {}
    rec = str((last or {}).get("baserate_reconciliation") or "").strip()
    if rec:
        lines.append(f"  reconciliation with the reference: {rec[:400]}")
    for slot, v in ((last or {}).get("plan") or {}).items():
        finding = str((v or {}).get("finding") or "").strip() if isinstance(v, dict) else ""
        if finding:
            lines.append(f"  plan {slot}: {finding[:200]}")
    return lines


def merged_ledger(trials: Sequence[Any]) -> List[Dict[str, Any]]:
    """The trials' ledgers merged: one item per (URL, quote), tagged with lanes."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str]] = []
    for t in trials:
        for it in getattr(t, "ledger", None) or []:
            key = (str(it.get("url") or "").strip(), str(it.get("quote") or "").strip())
            if not key[1]:
                continue
            if key not in merged:
                merged[key] = dict(it, lanes=[])
                order.append(key)
            lane = getattr(t, "lane", "") or "?"
            if lane not in merged[key]["lanes"]:
                merged[key]["lanes"].append(lane)
    return [merged[k] for k in order]


def _priority(item: Dict[str, Any]) -> Tuple[int, int]:
    try:
        tier = int(item.get("tier") or 5)
    except (TypeError, ValueError):
        tier = 5
    dated_figure = bool(item.get("date")) and bool(_DIGIT.search(str(item.get("quote") or "")))
    return (0 if tier <= 2 else 1, 0 if dated_figure else 1)


def _item_line(it: Dict[str, Any]) -> str:
    return (
        f"- [{'/'.join(it.get('lanes') or [])}] {it.get('date') or 'undated'}, tier "
        f"{it.get('tier') or '?'}, {it.get('kind') or 'unclassified'}, "
        f"{it.get('direction') or 'neutral'}: \"{str(it.get('quote') or '')[:300]}\" "
        f"({it.get('url') or 'no url'})"
    )


def dispute(trials: Sequence[Any], metric: str) -> Dict[str, Any]:
    """The disagreement in numbers: medians by lane per month, the largest
    pairwise month-1 JSD, and the buckets carrying the difference."""
    from pythia.buckets import labels_for  # noqa: PLC0415
    from sibyl.aggregate import month_vector  # noqa: PLC0415
    from sibyl.trials import max_pairwise_jsd, month_median  # noqa: PLC0415

    medians: Dict[str, Dict[str, float]] = {}
    for m in (1, 6):
        for t in trials:
            mb = t.month_beliefs.get(m)
            if mb is not None:
                medians.setdefault(f"month_{m}", {})[t.lane or str(t.trial_index)] = round(
                    month_median(mb.p_zero, mb.quantiles_positive), 1)
    vecs = [month_vector(t.month_beliefs[1].dist(), metric) for t in trials]
    spread = max_pairwise_jsd(vecs)
    buckets: List[Dict[str, Any]] = []
    if len(vecs) >= 2:
        try:
            labels = labels_for(metric)
        except Exception:  # noqa: BLE001
            labels = []
        ranges = []
        for b in range(len(vecs[0])):
            col = [v[b] for v in vecs]
            ranges.append((max(col) - min(col), b, min(col), max(col)))
        for gap, b, lo, hi in sorted(ranges, reverse=True)[:2]:
            buckets.append({"bucket": b + 1, "label": labels[b] if b < len(labels) else str(b + 1),
                            "low": round(lo, 3), "high": round(hi, 3), "gap": round(gap, 3)})
    return {"medians_by_lane": medians, "max_pairwise_jsd": spread, "buckets": buckets}


def build_brief(trials: Sequence[Any], metric: str,
                max_chars: Optional[int] = None) -> ReconcileBrief:
    """The lane R brief from the valid production trials."""
    max_chars = int(_cfg.RECONCILE_BRIEF_MAX_CHARS if max_chars is None else max_chars)
    disp = dispute(trials, metric)
    head: List[str] = ["=== THE EARLIER TRIALS AND WHERE THEY DIFFER ==="]
    for t in trials:
        head.extend(_trial_lines(t))
    head.append("The dispute, stated by the code:")
    for month, by_lane in disp["medians_by_lane"].items():
        head.append(f"  {month.replace('_', ' ')} medians by lane: "
                    + ", ".join(f"{lane} {_fmt(v)}" for lane, v in by_lane.items()))
    if disp["max_pairwise_jsd"] is not None:
        head.append(f"  largest pairwise month-1 Jensen-Shannon divergence: "
                    f"{disp['max_pairwise_jsd']:.3f}")
    for b in disp["buckets"]:
        head.append(f"  bucket {b['bucket']} ({b['label']}): the trials put between "
                    f"{b['low']:.2f} and {b['high']:.2f} on it")
    items = merged_ledger(trials)
    ranked = sorted(range(len(items)), key=lambda i: (_priority(items[i]), i))
    head_text = "\n".join(head)
    kept: List[int] = []
    used = len(head_text) + len("\nTheir evidence (lanes that recorded each item in brackets):") + 80
    for i in ranked:
        line = _item_line(items[i])
        if used + len(line) + 1 > max_chars:
            continue
        kept.append(i)
        used += len(line) + 1
    dropped = len(items) - len(kept)
    body = [head_text, "Their evidence (lanes that recorded each item in brackets):"]
    body.extend(_item_line(items[i]) for i in sorted(kept))
    if not kept:
        body.append("(none recorded)")
    if dropped:
        body.append(f"({dropped} ledger item(s) left out for length; tier 1-2 and dated "
                    "figures were kept first)")
    return ReconcileBrief(text="\n".join(body), dispute=disp, n_items=len(items),
                          n_items_dropped=dropped)


def median_inside(r_trial: Any, production: Sequence[Any]) -> Optional[bool]:
    """Did R's month-1 median fall inside the range of the production medians?"""
    from sibyl.trials import month_median  # noqa: PLC0415

    try:
        meds = [month_median(t.month_beliefs[1].p_zero, t.month_beliefs[1].quantiles_positive)
                for t in production]
        mb = r_trial.month_beliefs[1]
        r = month_median(mb.p_zero, mb.quantiles_positive)
    except (KeyError, AttributeError):
        return None
    if not meds:
        return None
    return min(meds) <= r <= max(meds)
