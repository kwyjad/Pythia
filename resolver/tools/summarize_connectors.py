"""Compat shim for the legacy resolver.tools import path.

This module re-exports the canonical connector summary helpers that live in
``scripts.ci.summarize_connectors`` so existing tooling can continue importing
``resolver.tools.summarize_connectors``.  The CI implementation already
contains the logic for rendering an em dash in the "Meta rows" column for
ok-empty/header-only runs, so re-exporting those functions keeps both code paths
aligned without duplicating the implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.ci import summarize_connectors as _ci

SUMMARY_TITLE = _ci.SUMMARY_TITLE
MISSING_REPORT_SUMMARY = _ci.MISSING_REPORT_SUMMARY


def load_report(path: Path) -> list[dict[str, Any]]:
    """Load connector report entries from *path*.

    This simply forwards to :func:`scripts.ci.summarize_connectors.load_report`
    so that both import paths return identical data structures.
    """

    return _ci.load_report(path)


def build_markdown(entries: Sequence[Mapping[str, Any]]) -> str:
    """Render a Markdown summary for *entries*.

    Delegates to the CI implementation, which already applies the display
    policy of showing an em dash for ok-empty/header-only connectors in the
    "Meta rows" column.
    """

    return _ci.build_markdown(entries)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point mirroring ``python -m scripts.ci.summarize_connectors``."""

    return _ci.main(argv)


__all__ = [
    "SUMMARY_TITLE",
    "MISSING_REPORT_SUMMARY",
    "load_report",
    "build_markdown",
    "main",
]
