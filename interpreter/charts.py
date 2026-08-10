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


def legend() -> str:
    """One line explaining what the bars are, printed once."""
    return (
        "Each bar is the chance the month lands in that size band. The darker "
        "bar is the most likely band. A long row of small bars to the right "
        "means the system is holding open the chance of something much larger."
    )
