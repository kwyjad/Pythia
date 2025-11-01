"""Shared helpers for ingestion connectors."""

from .error_report import read_log_tail, write_error_report
from .validation import validate_required_fields, write_json

__all__ = [
    "read_log_tail",
    "write_error_report",
    "validate_required_fields",
    "write_json",
]
