from __future__ import annotations

import json

import pytest

try:  # pragma: no cover - skip guard for environments without DuckDB
    import duckdb  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("duckdb not installed", allow_module_level=True)

from horizon_scanner import horizon_scanner as hs


def test_parse_hs_triage_json_plain() -> None:
    payload = {"hazards": {"FL": {"tier": "priority"}}}
    parsed = hs._parse_hs_triage_json(json.dumps(payload))
    assert parsed["hazards"]["FL"]["tier"] == "priority"


def test_parse_hs_triage_json_fenced() -> None:
    raw = "```json\n{\"hazards\": {\"DR\": {\"triage_score\": 0.5}}}\n```"
    parsed = hs._parse_hs_triage_json(raw)
    assert parsed["hazards"]["DR"]["triage_score"] == 0.5


def test_parse_hs_triage_json_invalid() -> None:
    with pytest.raises(json.JSONDecodeError):
        hs._parse_hs_triage_json("no json here")
