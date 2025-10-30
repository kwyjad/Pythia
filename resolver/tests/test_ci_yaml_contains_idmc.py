"""Ensure CI workflows wire in the IDMC connector."""
from __future__ import annotations

import pathlib
import re

import yaml

WF_MONTHLY = pathlib.Path(".github/workflows/resolver-monthly.yml")
WF_BACKFILL = pathlib.Path(".github/workflows/resolver-initial-backfill.yml")


def _load_yaml(path: pathlib.Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_monthly_mentions_idmc() -> None:
    content = WF_MONTHLY.read_text(encoding="utf-8")
    assert "Probe IDMC reachability" in content
    assert re.search(r"idmc", content, re.IGNORECASE)
    assert _load_yaml(WF_MONTHLY)  # basic YAML sanity


def test_backfill_mentions_idmc() -> None:
    content = WF_BACKFILL.read_text(encoding="utf-8")
    assert "Probe IDMC reachability" in content
    assert re.search(r"idmc", content, re.IGNORECASE)
    assert _load_yaml(WF_BACKFILL)
