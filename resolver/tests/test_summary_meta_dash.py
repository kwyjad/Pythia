# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from scripts.ci import summarize_connectors as summary


def test_zero_rows_render_as_dash() -> None:
    fmt = getattr(summary, "_fmt_count", None)
    assert fmt is not None, "expected _fmt_count helper to exist"
    assert fmt(0) == "—"
    assert fmt(None) == "—"
    assert fmt(5) == "5"
