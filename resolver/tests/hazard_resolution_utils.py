# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared helpers for the hazard-resolution test suites."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from resolver.hazard_resolution.rulebook import DEFAULT_RULEBOOK_PATH, Rulebook

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hazard_resolution"
IBTRACS_SAMPLE_CSV = FIXTURES / "ibtracs_sample.csv"
SYNTHETIC_COUNTRIES_GEOJSON = FIXTURES / "synthetic_countries.geojson"
RELIEFWEB_GOLDEN_SET = FIXTURES / "reliefweb_golden_set.json"


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def make_rulebook(overrides: dict[str, Any] | None = None) -> Rulebook:
    """The real rulebook.yaml, optionally with test overrides deep-merged."""
    with open(DEFAULT_RULEBOOK_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if overrides:
        data = _deep_merge(data, overrides)
    return Rulebook(data, DEFAULT_RULEBOOK_PATH)


def seed_ibtracs(con, scope: str = "last3years") -> dict:
    """Parse + store the fixture IBTrACS CSV into ``con``."""
    from resolver.hazard_resolution.ibtracs import parse_ibtracs_csv, store_ibtracs

    frame = parse_ibtracs_csv(IBTRACS_SAMPLE_CSV)
    return store_ibtracs(
        con, frame, scope, "https://example.test/ibtracs.last3years.list.v04r01.csv"
    )


def make_candidate(source: str, value: float, **overrides: Any):
    """A :class:`Candidate` on ``source`` with sensible defaults.

    Value type follows the source unless overridden, so a test cannot
    accidentally build an IDU candidate that claims to be an ``affected``
    figure — that mislabelling is exactly what the ladder tests guard.
    """
    from resolver.hazard_resolution.candidates import (
        VALUE_AFFECTED,
        VALUE_CEILING,
        VALUE_LOWER_BOUND,
        Candidate,
    )

    default_type = {
        "idmc_idu": VALUE_LOWER_BOUND,
        "gdacs": VALUE_CEILING,
    }.get(source, VALUE_AFFECTED)

    fields: dict[str, Any] = {
        "iso3": "PHL",
        "ym": "2024-03",
        "hazard": "FL",
        "value": float(value),
        "value_type": default_type,
        "source": source,
        "source_ref": f"{source}-ref-1",
        "stated_by": source,
        "doc_url": f"https://example.test/{source}/1",
        "span_start": "2024-03-05",
        "span_end": "2024-03-09",
        "retrieved_at": "2024-04-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return Candidate(**fields)


def seed_population(con, iso3: str = "PHL", year: int = 2024, population: int = 115_000_000):
    """A national population denominator for the sanity cap."""
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    con.execute(
        """
        INSERT OR IGNORE INTO haz_raw_population
            (iso3, year, population, source, product, source_url)
        VALUES (?, ?, ?, 'test', 'fixture', 'https://example.test/pop')
        """,
        [iso3, year, population],
    )


def seed_gdacs_event(
    con,
    *,
    iso3: str = "PHL",
    ym: str = "2024-03",
    hazard: str = "FL",
    event_id: str = "1000001",
    alert_level: str = "Orange",
    exposed_population: float = 500_000.0,
    start_date: str = "2024-03-05",
    end_date: str = "2024-03-09",
    months_overlapped: list[str] | None = None,
):
    """One GDACS event in ``haz_raw_gdacs`` (detection input + ceiling)."""
    from resolver.hazard_resolution.rules import event_months
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    import datetime as dt

    overlapped = months_overlapped or event_months(
        dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    )
    record = RawRecord(
        record_id=f"{hazard}-{event_id}",
        payload={
            "event_id": event_id,
            "event_type": hazard,
            "hazard": hazard,
            "iso3_list": [iso3],
            "country": iso3,
            "alert_level": alert_level,
            "alert_score": 1.5,
            "exposed_population": exposed_population,
            "start_date": start_date,
            "end_date": end_date,
            "months_overlapped": overlapped,
            "published_at": end_date,
            "fetched_at": "2024-04-01T00:00:00+00:00",
        },
        iso3=iso3,
        ym=ym,
        hazard=hazard,
        source_url=f"https://example.test/gdacs/{event_id}",
    )
    return store_raw_records(con, "gdacs", [record])


def silent_sweep_evidence(iso3: str, ym: str) -> dict[str, Any]:
    """A well-formed silent ReliefWeb sweep record (as the sweep returns)."""
    return {
        "iso3": iso3,
        "ym": ym,
        "hazard_key": "cyclone",
        "silent": True,
        "inconclusive": False,
        "total_hits": 0,
        "queries": [
            {
                "kind": "disaster_type",
                "url": "https://api.reliefweb.int/v2/reports",
                "payload": {"filter": {"conditions": ["…"]}},
                "hits": 0,
                "sample": [],
            },
            {
                "kind": "keywords",
                "url": "https://api.reliefweb.int/v2/reports",
                "payload": {"query": {"value": "cyclone OR hurricane"}},
                "hits": 0,
                "sample": [],
            },
        ],
        "retrieved_at": "2026-01-15T00:00:00+00:00",
        "error": None,
    }


def load_golden_set() -> dict[str, Any]:
    """The ReliefWeb extraction golden set (documents + expected output)."""
    with open(RELIEFWEB_GOLDEN_SET, encoding="utf-8") as fh:
        return json.load(fh)


def golden_documents() -> list[dict[str, Any]]:
    """Just the ``doc`` payloads, in fixture order."""
    return [case["doc"] for case in load_golden_set()["documents"]]


def seed_reliefweb_docs(
    con,
    documents: list[dict[str, Any]],
    *,
    iso3: str = "PHL",
    ym: str = "2024-03",
    hazard: str = "FL",
):
    """Store documents in ``haz_raw_reliefweb_docs`` as the connector would."""
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    records = [
        RawRecord(
            record_id=f"rw-{doc['doc_id']}",
            payload={**doc, "iso3": iso3, "ym": ym, "hazard": hazard},
            iso3=iso3,
            ym=ym,
            hazard=hazard,
            source_url=doc.get("url"),
        )
        for doc in documents
    ]
    return store_raw_records(con, "reliefweb_docs", records)


def golden_call_fn(documents: list[dict[str, Any]] | None = None):
    """A model seam that replays the golden set's recorded responses.

    Returns ``(call, calls)`` where ``calls`` is the list of prompts sent —
    so a test can assert not only what came back but that nothing was
    called at all.
    """
    cases = {case["doc"]["doc_id"]: case for case in load_golden_set()["documents"]}
    calls: list[str] = []

    def call(model_ref: str, prompt: str, rulebook) -> tuple[str, dict, str]:
        calls.append(prompt)
        for doc_id, case in cases.items():
            # The prompt embeds the document's URL, which is unique per case.
            if case["doc"]["url"] in prompt:
                return case["response"], {"prompt_tokens": 6000, "completion_tokens": 200}, ""
        return '{"figures": []}', {"prompt_tokens": 10, "completion_tokens": 5}, ""

    return call, calls
