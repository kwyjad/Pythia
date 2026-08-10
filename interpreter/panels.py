# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the reader is told about how the report was made.

A reader must be able to see why something is in the report and why something
else is not. Three pieces, all GENERATED from the configuration and the gate's
own counts rather than written by the model: a panel stating the rules, a tag
on every entry naming the gate it passed, and an appendix table listing every
question considered.

The counts come from one place (`gating.gate_rows`) so the panel, the tags and
the table cannot disagree. Three numbers that ought to match and are computed
three times will eventually not match, and the reader has no way to tell which
is right.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from interpreter import config, gating, names
from pythia.buckets import labels_for, thresholds_for


def humanise_bucket_label(label: str, metric: str, index: int) -> str:
    """"1-<10k" is a machine's label. Readers get "1 to 9,999 people"."""
    unit = "people"
    if (metric or "").upper() == "FATALITIES":
        unit = "deaths"
    edges = thresholds_for(metric)
    if not edges or index + 1 >= len(edges):
        return str(label)
    lo, hi = float(edges[index]), float(edges[index + 1])
    if lo == 0 and hi <= 1:
        return f"no {unit} recorded"
    if math.isinf(hi):
        return f"{lo:,.0f} {unit} or more"
    if hi - lo == 1:
        return f"{lo:,.0f} {unit}"
    return f"{lo:,.0f} to {hi - 1:,.0f} {unit}"


def humanised_labels(metric: str) -> list[str]:
    return [
        humanise_bucket_label(lbl, metric, i)
        for i, lbl in enumerate(labels_for(metric) or [])
    ]


def _threshold_sentence(metric: str) -> str | None:
    """One line of the panel: what counts as material for this metric."""
    absolute = config.threshold_for_metric(metric)
    if absolute is None:
        return None
    share = config.population_share_for_metric(metric)
    name = names.metric_name(metric, short=True)
    unit = "deaths" if metric.upper() == "FATALITIES" else "people"
    text = f"{name}: {absolute:,.0f} {unit}"
    if share:
        text += f", or {share * 100:.0f}% of the country's population, whichever is lower"
    return text


def selection_panel(counts: dict[str, Any]) -> dict[str, Any]:
    """The "How these entries were chosen" panel, as structured data.

    Returned as data rather than prose so the markdown renderer, the PDF and
    the dashboard each lay it out their own way while stating one set of
    facts. Every number here comes from the gate's own counts.
    """
    lines = [
        s for s in (
            _threshold_sentence("PA"),
            _threshold_sentence("FATALITIES"),
            _threshold_sentence("PHASE3PLUS_IN_NEED"),
        ) if s
    ]
    min_prob = config.min_probability()
    pct = config.unusual_percentile()
    return {
        "title": "How these entries were chosen",
        "ordering": (
            "Entries are ordered by how many ADDITIONAL people are expected "
            "beyond what history would suggest, not by how unusual the "
            "forecast looks. A forecast can be many times its usual level and "
            "still concern very few people."
        ),
        "tests": [
            (
                f"Unusual: the forecast departs from its own history by more "
                f"than {pct * 100:.0f}% of this month's other forecasts do."
            ),
            (
                f"Material: at least a {min_prob * 100:.0f}% chance, in some "
                f"month of the window, of passing a size worth mobilising for."
            ),
        ],
        "thresholds": lines,
        "counts": {
            "considered": counts.get("considered", 0),
            "cleared_both": counts.get("both", 0),
            "heavy_burden": counts.get("major", 0),
            "watchlist": counts.get("watchlist", 0),
            "thin_anchor": counts.get("thin", 0),
        },
        "counts_sentence": (
            f"{counts.get('considered', 0)} forecasts were considered. "
            f"{counts.get('both', 0)} cleared both tests. "
            f"{counts.get('major', 0)} carry a heavy burden without a call "
            f"that it is worsening. "
            f"{counts.get('watchlist', 0)} are unusual but too small to act "
            f"on, and are held on the watchlist. "
            # Without this a reader who sees a dozen cleared entries and five
            # printed ones concludes the report lost seven of them.
            f"The report carries at most {config.max_entries()} entries, so "
            f"the rest appear only in the table at the back. "
            # The gate emits "thin"; the panel renames it "thin_anchor" for the
            # reader. Reading the reader-facing name back out of the gate's
            # dict silently printed zero.
            f"{counts.get('thin', 0)} rest on a thin historical record."
        ),
        "thin_note": (
            "A thin record means fewer than "
            f"{config.baserate_min_obs()} historical observations behind the "
            "comparison. Those entries are ranked below the rest and say so, "
            "because a large multiple of almost nothing is not evidence of "
            "much."
        ),
    }


_TABLE_FIELDS = (
    "country", "hazard", "metric", "excess", "chance", "gate", "anchor",
)


def question_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Every question considered, one row each: the audit trail.

    This is the thing that answers "why is my country not in the report",
    which is the question the selection rules will actually be asked.
    """
    out: list[dict[str, str]] = []
    for row in sorted(rows, key=gating.rank_key):
        excess = row.get("excess_nominal")
        exceedances = row.get("exceedances") or []
        best = max(exceedances) if exceedances else None
        out.append({
            "country": names.country_name(row.get("iso3")),
            "hazard": names.hazard_name(row.get("hazard_code")),
            "metric": names.metric_name(row.get("metric"), short=True),
            "excess": (
                f"{excess:+,.0f}" if isinstance(excess, (int, float)) else "n/a"
            ),
            "chance": f"{best * 100:.0f}%" if best is not None else "n/a",
            "gate": str(row.get("gate") or "not selected"),
            "anchor": "thin" if row.get("baserate_thin") else "",
        })
    return out


def planning_figures(
    monthly_probs: Sequence[Sequence[float]],
    metric: str,
    month_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The two figures a response planner can act on, plus the caveat.

    A bucket probability table is a picture of uncertainty and nobody can plan
    against a picture. p50 is the planning figure, p90 the contingency
    figure, and p_zero the chance the month passes with nothing recorded.
    Reported at the PEAK horizon, because that is the month the plan has to
    cover.
    """
    if not monthly_probs:
        return {}
    evs = [gating.expected_value(m, metric) for m in monthly_probs]
    peak = gating.peak_horizon(evs) or 1
    probs = monthly_probs[peak - 1]
    month = None
    if month_labels and peak - 1 < len(month_labels):
        month = month_labels[peak - 1]
    return {
        "peak_horizon": peak,
        "peak_month": month,
        "p50_peak": gating.quantile(probs, metric, 0.5),
        "p90_peak": gating.quantile(probs, metric, 0.9),
        "p_zero_peak": gating.p_zero(probs),
        # Whether the window is flat or spiked, for the clause that follows
        # the two figures.
        "window_shape": _window_shape(evs),
    }


def _window_shape(evs: Sequence[float]) -> str:
    """A clause describing the six months, not a number."""
    if not evs or len(evs) < 2:
        return "a single month"
    top = max(evs)
    if top <= 0:
        return "flat across the window"
    others = sorted(evs, reverse=True)[1:]
    second = others[0] if others else 0.0
    if second >= 0.9 * top:
        return "steady across the window"
    if second <= 0.4 * top:
        return "concentrated in that month"
    return "rising towards that month"
