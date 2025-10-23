"""Shared helpers for ingestion connectors."""

from .validation import validate_required_fields, write_json

__all__ = ["validate_required_fields", "write_json"]
