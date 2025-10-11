"""Resolver diagnostics helpers gated behind the ``RESOLVER_DIAG`` flag."""

from __future__ import annotations

from .diagnostics import diag_enabled, dump_counts, dump_table_meta, get_logger, log_json

__all__ = [
    "diag_enabled",
    "dump_counts",
    "dump_table_meta",
    "get_logger",
    "log_json",
]
