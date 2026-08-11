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
from typing import Any, Mapping, Sequence

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


# A count of one takes a singular verb. "1 were both unusual and moved by
# enough" is the kind of slip a reader takes as carelessness about everything
# else in the sentence.
def _were(n: int) -> str:
    return "was" if n == 1 else "were"


def _are(n: int) -> str:
    return "is" if n == 1 else "are"


def _carry(n: int) -> str:
    return "carries" if n == 1 else "carry"


def _rest(n: int) -> str:
    return "rests" if n == 1 else "rest"


def selection_panel(counts: dict[str, Any]) -> dict[str, Any]:
    """The "How these entries were chosen" panel, as structured data.

    Returned as data rather than prose so the markdown renderer, the PDF and
    the dashboard each lay it out their own way while stating one set of
    facts. Every number here comes from the gate's own counts, and the tests
    are described from the mode actually in force: a panel that describes last
    month's rule is confidently wrong, which is worse than one describing none.
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
    delta_mode = str(counts.get("mode") or config.gate_mode()) != gating.MODE_LEVEL
    if delta_mode:
        worsening_test = (
            "Worsening: at the month with most expected impact, either the "
            "figure you would plan against or the figure you would hold "
            "contingency for has RISEN above its historical equivalent by at "
            "least the size below. Testing the level instead of the movement "
            "put a country with a permanently large caseload in this section "
            "every month."
        )
    else:
        worsening_test = (
            f"Worsening: at least a {min_prob * 100:.0f}% chance, in some "
            "month of the window, of passing a size worth mobilising for, "
            "with more people expected than history would suggest."
        )
    return {
        "title": "How these entries were chosen",
        "ordering": (
            "The two sections are ordered differently, and each heading says "
            "how. Worsening situations are ordered by how far the planning "
            "figures have risen. Heavy burdens are ordered by the expected "
            "number of people affected. A forecast can be many times its "
            "usual level and still concern very few people, which is why "
            "neither section is ordered by a multiple."
        ),
        "tests": [
            (
                f"Unusual: the forecast departs from its own history by more "
                f"than {pct * 100:.0f}% of this month's other forecasts do."
            ),
            worsening_test,
            (
                f"Heavy burden: at least a {min_prob * 100:.0f}% chance, in "
                "some month of the window, of passing that same size, whether "
                "or not anything has changed. These are the places already in "
                "trouble."
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
            f"{counts.get('both', 0)} "
            f"{_were(counts.get('both', 0))} both unusual and moved by enough. "
            f"{counts.get('major', 0)} {_carry(counts.get('major', 0))} a "
            f"heavy burden without a call that it is worsening. "
            f"{counts.get('watchlist', 0)} "
            f"{_are(counts.get('watchlist', 0))} unusual but too small to act "
            f"on, and held on the watchlist. "
            # Without this a reader who sees a dozen cleared entries and five
            # printed ones concludes the report lost seven of them.
            f"The report carries at most {config.max_entries()} entries, so "
            f"the rest appear only in the table at the back. "
            # The gate emits "thin"; the panel renames it "thin_anchor" for the
            # reader. Reading the reader-facing name back out of the gate's
            # dict silently printed zero.
            f"{counts.get('thin', 0)} "
            f"{_rest(counts.get('thin', 0))} on a thin historical record."
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
    "country", "hazard", "metric", "excess", "move_p50", "move_p90",
    "gate", "anchor",
)

# Why a movement cell can legitimately be empty. A blank or an "n/a" makes the
# transparency table decorative and invites the reader to distrust the rest of
# it, so every cell that cannot carry a number carries the reason instead.
_NO_MOVEMENT_REASONS = {
    "binary": "yes/no question",
    "no_anchor": "no historical anchor",
    "no_peak": "no forecast at the peak month",
}


def _movement_cell(
    value: Any, *, metric: str, score_family: str, has_anchor: bool, side: str
) -> str:
    """One movement cell: a signed figure in the metric's own units, or a
    stated reason. Never a bare "n/a"."""
    if isinstance(value, (int, float)):
        if str(score_family or "").lower() == "binary":
            return f"{float(value) * 100:+.0f} points"
        unit = "deaths" if str(metric or "").upper() == "FATALITIES" else "people"
        return f"{_signed_round(float(value))} {unit}"
    if str(score_family or "").lower() == "binary":
        return (
            _NO_MOVEMENT_REASONS["binary"] if side == "p90"
            else _NO_MOVEMENT_REASONS["no_peak"]
        )
    if not has_anchor:
        return _NO_MOVEMENT_REASONS["no_anchor"]
    return _NO_MOVEMENT_REASONS["no_peak"]


def _signed_round(value: float) -> str:
    """A movement rounded to its own scale, with its sign kept.

    The figure comes from interpolating inside a bucket whose width is tens of
    thousands. Printing "+9,577,500" claims a precision the bucket scheme does
    not have, and a reader remembers the digits.
    """
    magnitude = abs(value)
    sign = "+" if value >= 0 else "-"
    if magnitude >= 1_000_000:
        return f"{sign}{magnitude / 1_000_000:.1f}m".replace(".0m", "m")
    if magnitude >= 1_000:
        return f"{sign}{round(magnitude / 1_000):,.0f},000"
    return f"{sign}{magnitude:,.0f}"


def question_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Every question considered, one row each: the audit trail.

    This is the thing that answers "why is my country not in the report",
    which is the question the selection rules will actually be asked. The two
    movement columns are the evidence for the gate, so they are populated for
    every row: where a movement genuinely cannot be computed the cell says
    why, because a column of "n/a" is a column a reader learns to skip.
    """
    out: list[dict[str, str]] = []
    for row in sorted(rows, key=gating.rank_key):
        excess = row.get("excess_nominal")
        metric = str(row.get("metric") or "")
        family = str(row.get("score_family") or "")
        has_anchor = row.get("movement_threshold") is not None
        out.append({
            "country": names.country_name(row.get("iso3")),
            "hazard": names.hazard_name(row.get("hazard_code")),
            "metric": names.metric_name(metric, short=True),
            "excess": (
                _signed_round(float(excess))
                if isinstance(excess, (int, float)) else "no anchor"
            ),
            "move_p50": _movement_cell(
                row.get("delta_p50"), metric=metric, score_family=family,
                has_anchor=has_anchor, side="p50",
            ),
            "move_p90": _movement_cell(
                row.get("delta_p90"), metric=metric, score_family=family,
                has_anchor=has_anchor, side="p90",
            ),
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


def humanise_count(value: float, metric: str) -> str:
    """A count a planner would actually write on a form.

    "About 9.6 million people", never "+9,577,500 people". The figure is an
    interpolation inside a bucket tens of thousands wide, so every digit past
    the third is a claim the bucket scheme cannot support, and a reader keeps
    the digits they are shown.
    """
    unit = "deaths" if (metric or "").upper() == "FATALITIES" else "people"
    v = abs(float(value))
    if v < 1:
        return f"almost no {unit}"
    if v >= 1_000_000:
        return f"about {v / 1_000_000:.1f} million {unit}".replace(".0 million", " million")
    if v >= 1_000:
        magnitude = 10 ** max(0, int(math.floor(math.log10(v))) - 2)
        return f"about {round(v / magnitude) * magnitude:,.0f} {unit}"
    return f"about {round(v):,.0f} {unit}"


def _terminal_bucket_floor(value: float | None, metric: str) -> float | None:
    """The lower bound of the open-ended top band, when a figure lands in it.

    The top band has no upper edge, so the quantile function returns its
    centroid for anything inside it. Two quantiles that both land there print
    the same number, which is how Sudan came to read "Plan against about
    20,000,000 people. Hold contingency for 20,000,000 people."
    """
    if value is None:
        return None
    edges = thresholds_for(metric)
    if not edges or len(edges) < 2:
        return None
    floor = float(edges[-2])
    return floor if float(value) >= floor else None


def planning_sentence(row: Mapping[str, Any]) -> str | None:
    """The two figures a planner acts on, as a finished sentence.

    Generated rather than written. It is a frame with two numbers in it and no
    judgement, and leaving it to the model produced "+5 people more deaths"
    and a contingency figure identical to the planning figure. Returns None
    when there is nothing to plan against, so the entry simply omits the line
    rather than printing a hole.
    """
    metric = str(row.get("metric") or "")
    month = row.get("peak_month")
    when = f" in {month}" if month else ""
    if str(row.get("score_family") or "").lower() == "binary":
        # A yes/no question has no size to plan against, so it gets no
        # planning figure rather than a made-up one.
        return None

    p50, p90 = row.get("p50_peak"), row.get("p90_peak")
    if p50 is None and p90 is None:
        return None

    # Both quantiles inside the open-ended top band: one sentence using the
    # band's lower bound, and no contingency line to contradict it.
    floor50 = _terminal_bucket_floor(p50, metric)
    floor90 = _terminal_bucket_floor(p90, metric)
    if floor50 is not None and floor90 is not None and floor50 == floor90:
        unit = "deaths" if metric.upper() == "FATALITIES" else "people"
        return (
            f"Plan against at least {_round_words(floor50)} {unit}{when}. "
            "The forecast puts both the middle and the upper end of its range "
            "in the top band, which has no ceiling."
        )

    parts: list[str] = []
    if p50 is not None:
        parts.append(f"Plan against {humanise_count(p50, metric)}{when}.")
    if p90 is not None:
        parts.append(f"Hold contingency for {humanise_count(p90, metric)}.")
    p_zero = row.get("p_zero_peak")
    if isinstance(p_zero, (int, float)) and float(p_zero) >= 0.2:
        parts.append(
            f"There is a {float(p_zero) * 100:.0f}% chance of nothing recorded "
            "at all."
        )
    return " ".join(parts) or None


def _round_words(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f} million"
    if value >= 1_000:
        return f"{value / 1_000:.0f},000"
    return f"{value:,.0f}"


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
