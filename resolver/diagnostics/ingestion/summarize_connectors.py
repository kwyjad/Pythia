"""Diagnostics shim that delegates to the CI summarizer implementation."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

# This module delegates to the CI summarizer to preserve single-source legacy behavior.
from scripts.ci import summarize_connectors as _ci

SUMMARY_PATH = _ci.SUMMARY_PATH
SUMMARY_TITLE = _ci.SUMMARY_TITLE
LEGACY_TITLE = _ci.LEGACY_TITLE
DEFAULT_DIAG_DIR = _ci.DEFAULT_DIAG_DIR
DEFAULT_STAGING_DIR = _ci.DEFAULT_STAGING_DIR


def load_report(path: os.PathLike[str] | str) -> list[Mapping[str, Any]]:
    """Proxy to :func:`scripts.ci.summarize_connectors.load_report`."""

    return _ci.load_report(path)


def build_markdown(
    entries: Sequence[Mapping[str, Any]] | None,
    diagnostics_root: os.PathLike[str] | str = DEFAULT_DIAG_DIR,
    staging_root: os.PathLike[str] | str = DEFAULT_STAGING_DIR,
) -> str:
    """Render markdown via the CI summarizer implementation."""

    return _ci.build_markdown(entries, diagnostics_root=diagnostics_root, staging_root=staging_root)


def render_summary_md(
    entries: Sequence[Mapping[str, Any]] | None,
    *,
    diagnostics_root: os.PathLike[str] | str = DEFAULT_DIAG_DIR,
    staging_root: os.PathLike[str] | str = DEFAULT_STAGING_DIR,
    output_path: os.PathLike[str] | str | None = None,
) -> str:
    """Proxy to :func:`scripts.ci.summarize_connectors.render_summary_md`."""

    return _ci.render_summary_md(
        entries,
        diagnostics_root=diagnostics_root,
        staging_root=staging_root,
        output_path=output_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point delegating to :func:`scripts.ci.summarize_connectors.main`."""

    return _ci.main(argv)


__all__ = [
    "SUMMARY_PATH",
    "SUMMARY_TITLE",
    "LEGACY_TITLE",
    "DEFAULT_DIAG_DIR",
    "DEFAULT_STAGING_DIR",
    "load_report",
    "build_markdown",
    "render_summary_md",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
