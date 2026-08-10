# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Small inline SVG graphics for the report.

A forecast here is a distribution over size bands, and a bar per band is the
one picture that shows what a sentence struggles to: whether the weight sits
in one place, or trails off into a long tail. Drawn as inline SVG so the PDF
needs no plotting library and no network, and so the image cannot disagree
with the numbers beside it.

Fred's palette (web/tailwind.config.ts) so the report looks like the rest of
the system.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

FRED_PRIMARY = "#156082"
FRED_SECONDARY = "#80350E"
FRED_TEXT = "#3A3A3A"
FRED_BORDER = "#D6D6D6"
FRED_MUTED = "#6B7280"

_BAR_HEIGHT = 15
_BAR_GAP = 5
_LABEL_WIDTH = 128
_TRACK_WIDTH = 210
_VALUE_WIDTH = 46


def _as_probs(spd: Any) -> list[float] | None:
    """Accept a list, or a {bucket_index: p} mapping, or nothing."""
    if isinstance(spd, dict):
        try:
            items = sorted(((int(k), float(v)) for k, v in spd.items()))
        except (TypeError, ValueError):
            return None
        values = [v for _, v in items]
    elif isinstance(spd, (list, tuple)):
        try:
            values = [float(v) for v in spd]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not values or any(v < 0 for v in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    return [v / total for v in values]


def _labels_for(bucket_labels: Any, n: int) -> list[str]:
    if isinstance(bucket_labels, dict):
        out: list[str] = []
        for i in range(1, n + 1):
            out.append(str(bucket_labels.get(str(i)) or bucket_labels.get(i) or f"band {i}"))
        return out
    if isinstance(bucket_labels, (list, tuple)) and len(bucket_labels) >= n:
        return [str(x) for x in bucket_labels[:n]]
    return [f"band {i}" for i in range(1, n + 1)]


def probability_chart(spd: Any, bucket_labels: Any = None, *, title: str = "") -> str:
    """A horizontal bar per size band. Empty string when there is nothing to
    draw, so a caller can append it unconditionally."""
    probs = _as_probs(spd)
    if not probs:
        return ""
    labels = _labels_for(bucket_labels, len(probs))
    top = max(probs)
    width = _LABEL_WIDTH + _TRACK_WIDTH + _VALUE_WIDTH
    header = 16 if title else 0
    height = header + len(probs) * (_BAR_HEIGHT + _BAR_GAP) + 4

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Forecast probability by size band">'
    ]
    if title:
        parts.append(
            f'<text x="0" y="11" font-family="sans-serif" font-size="10" '
            f'fill="{FRED_MUTED}">{html.escape(title)}</text>'
        )
    for i, (p, label) in enumerate(zip(probs, labels)):
        y = header + i * (_BAR_HEIGHT + _BAR_GAP)
        bar = max(1.0, (p / top) * _TRACK_WIDTH) if top > 0 else 1.0
        # The most likely band is picked out; the rest recede, so the eye
        # lands on where the weight actually is.
        fill = FRED_PRIMARY if p == top else "#9DC3D4"
        parts.append(
            f'<text x="0" y="{y + 11}" font-family="sans-serif" font-size="9" '
            f'fill="{FRED_TEXT}">{html.escape(label[:22])}</text>'
        )
        parts.append(
            f'<rect x="{_LABEL_WIDTH}" y="{y}" width="{_TRACK_WIDTH}" '
            f'height="{_BAR_HEIGHT}" fill="#F1F5F7" />'
        )
        parts.append(
            f'<rect x="{_LABEL_WIDTH}" y="{y}" width="{bar:.1f}" '
            f'height="{_BAR_HEIGHT}" fill="{fill}" />'
        )
        parts.append(
            f'<text x="{_LABEL_WIDTH + _TRACK_WIDTH + 6}" y="{y + 11}" '
            f'font-family="sans-serif" font-size="9" fill="{FRED_MUTED}">'
            f'{p * 100:.0f}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def probability_bar(p: float | None) -> str:
    """A single filled bar for a yes/no forecast."""
    if p is None:
        return ""
    try:
        value = max(0.0, min(float(p), 1.0))
    except (TypeError, ValueError):
        return ""
    width, height = 240, 16
    filled = value * (width - 60)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Chance of the event">'
        f'<rect x="0" y="0" width="{width - 60}" height="{height}" fill="#F1F5F7" />'
        f'<rect x="0" y="0" width="{filled:.1f}" height="{height}" fill="{FRED_PRIMARY}" />'
        f'<text x="{width - 54}" y="12" font-family="sans-serif" font-size="10" '
        f'fill="{FRED_TEXT}">{value * 100:.0f}%</text>'
        "</svg>"
    )


_DUMBBELL_LABEL_WIDTH = 150
_DUMBBELL_TRACK_WIDTH = 250
_DUMBBELL_ROW_HEIGHT = 18


def second_opinion_chart(rows: Sequence[dict[str, Any]], *, title: str = "") -> str:
    """Fred against the second reader, one row per covered question.

    Drawn on a SHARE axis rather than on the raw expected values: the covered
    questions mix deaths with millions of people in need, and one linear axis
    across both would say nothing. Each row plots Sibyl's share of the two
    expected values, so the centre line is agreement, right of it means Sibyl
    expects more and left of it means it expects less. That is dimensionless,
    so the rows are comparable, and it is the same dumbbell grammar the
    Sibyl performance page uses (which is React, and a PDF cannot run React).
    """
    plotted: list[tuple[str, float, float, float]] = []
    for row in rows:
        fred = row.get("fred_expected")
        sibyl = row.get("sibyl_expected")
        try:
            f, s = float(fred), float(sibyl)
        except (TypeError, ValueError):
            continue
        total = f + s
        if total <= 0:
            continue
        label = f"{row.get('country_name') or ''}, {str(row.get('hazard_name') or '').lower()}"
        plotted.append((label.strip(", "), f / total, s / total, total))
    if not plotted:
        return ""

    width = _DUMBBELL_LABEL_WIDTH + _DUMBBELL_TRACK_WIDTH + 20
    header = 16 if title else 0
    height = header + len(plotted) * _DUMBBELL_ROW_HEIGHT + 22
    x0 = _DUMBBELL_LABEL_WIDTH
    mid = x0 + _DUMBBELL_TRACK_WIDTH / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Where the second reader differs from the main system">'
    ]
    if title:
        parts.append(
            f'<text x="0" y="11" font-family="sans-serif" font-size="10" '
            f'fill="{FRED_MUTED}">{html.escape(title)}</text>'
        )
    parts.append(
        f'<line x1="{mid:.1f}" y1="{header}" x2="{mid:.1f}" '
        f'y2="{header + len(plotted) * _DUMBBELL_ROW_HEIGHT}" '
        f'stroke="{FRED_BORDER}" stroke-width="1" stroke-dasharray="2 2" />'
    )
    for i, (label, fred_share, sibyl_share, _total) in enumerate(plotted):
        y = header + i * _DUMBBELL_ROW_HEIGHT + _DUMBBELL_ROW_HEIGHT / 2
        fx = x0 + fred_share * _DUMBBELL_TRACK_WIDTH
        sx = x0 + sibyl_share * _DUMBBELL_TRACK_WIDTH
        parts.append(
            f'<text x="0" y="{y + 3:.1f}" font-family="sans-serif" '
            f'font-size="9" fill="{FRED_TEXT}">{html.escape(label[:30])}</text>'
        )
        parts.append(
            f'<line x1="{min(fx, sx):.1f}" y1="{y:.1f}" x2="{max(fx, sx):.1f}" '
            f'y2="{y:.1f}" stroke="{FRED_BORDER}" stroke-width="2" />'
        )
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{y:.1f}" r="3.5" fill="{FRED_PRIMARY}" />'
        )
        parts.append(
            f'<circle cx="{sx:.1f}" cy="{y:.1f}" r="3.5" fill="{FRED_SECONDARY}" />'
        )
    legend_y = header + len(plotted) * _DUMBBELL_ROW_HEIGHT + 14
    parts.append(
        f'<circle cx="{x0 + 4}" cy="{legend_y - 3:.0f}" r="3.5" fill="{FRED_PRIMARY}" />'
        f'<text x="{x0 + 12}" y="{legend_y:.0f}" font-family="sans-serif" '
        f'font-size="9" fill="{FRED_MUTED}">the main system</text>'
        f'<circle cx="{x0 + 110}" cy="{legend_y - 3:.0f}" r="3.5" fill="{FRED_SECONDARY}" />'
        f'<text x="{x0 + 118}" y="{legend_y:.0f}" font-family="sans-serif" '
        f'font-size="9" fill="{FRED_MUTED}">the second reader</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def second_opinion_legend() -> str:
    """One line explaining the share axis, printed under the chart."""
    return (
        "Each row places the two readers on a shared line. The dashed centre "
        "is agreement. A mark to the right of it means that reader expects "
        "more than the other. The scale is a share rather than a count, so "
        "rows about deaths and rows about millions of people can sit on one "
        "picture."
    )


def legend() -> str:
    """One line explaining what the bars are, printed once."""
    return (
        "Each bar is the chance the month lands in that size band. The darker "
        "bar is the most likely band. A long row of small bars to the right "
        "means the system is holding open the chance of something much larger."
    )
