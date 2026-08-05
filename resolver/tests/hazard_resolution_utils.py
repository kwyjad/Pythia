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


def seed_ipc_analysis(
    con,
    *,
    iso3: str,
    window_start: str,
    window_end: str,
    value: float,
    path: str = "ipc_api",
    analysis_date: str | None = None,
):
    """One IPC current-period analysis in ``haz_raw_ipc``.

    Written through the connector's own record builder so the cached shape
    is the one the drought rule reads in production — a test that seeded a
    hand-built payload could pass against a cache the connector never
    produces.
    """
    import datetime as dt

    from resolver.hazard_resolution.ipc import SOURCE, _analysis_record
    from resolver.hazard_resolution.sources import store_raw_records

    record = _analysis_record(
        iso3=iso3.upper(),
        path=path,
        window_start=dt.date.fromisoformat(window_start),
        window_end=dt.date.fromisoformat(window_end),
        value=float(value),
        analysis_date=analysis_date or window_start,
        source_url=f"https://example.test/ipc/{iso3.lower()}/{window_start}",
    )
    return store_raw_records(con, SOURCE, [record])


def seed_indicator_snapshot(
    con,
    *,
    name: str = "asap",
    provider: str = "asap",
    observed_ym: str = "2024-03",
    values: dict[str, Any] | None = None,
    url: str = "https://example.test/asap.json",
):
    """One cached drought-indicator feed snapshot.

    ``values`` maps ISO3 to whatever the feed states — a class label for
    ASAP, a number for an anomaly feed. Thresholds are applied at
    evaluation time, so the snapshot carries no verdict.
    """
    from resolver.hazard_resolution.drought_indicators import SOURCE
    from resolver.hazard_resolution.sources import RawRecord, store_raw_records

    payload_values = dict(values or {})
    record = RawRecord(
        record_id=f"{name}-{observed_ym}",
        payload={
            "name": name,
            "provider": provider,
            "url": url,
            "values": payload_values,
            "observed_ym": observed_ym,
            "observed_ym_source": "feed",
            "n_records": len(payload_values),
            "n_countries": len(payload_values),
            "n_unresolved": 0,
            "fetched_at": "2024-04-01T00:00:00+00:00",
        },
        ym=observed_ym,
        hazard="DR",
        source_url=url,
    )
    return store_raw_records(con, SOURCE, [record])


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


def seed_trigger(
    con,
    *,
    iso3: str,
    ym: str,
    hazard: str = "FL",
    triggered: bool = False,
    run_type: str = "live",
    trigger_source: str | None = None,
):
    """One ``haz_triggers`` row — the unit the resolution rate divides by."""
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    year, month = (int(p) for p in ym.split("-"))
    con.execute(
        """
        INSERT OR REPLACE INTO haz_triggers
            (iso3, year, month, hazard, triggered, trigger_source,
             trigger_detail_json, evidence_of_absence_json, run_type)
        VALUES (?, ?, ?, ?, ?, ?, '{}', NULL, ?)
        """,
        [iso3.upper(), year, month, hazard, triggered, trigger_source, run_type],
    )


def seed_resolution(
    con,
    *,
    iso3: str,
    ym: str,
    hazard: str = "FL",
    status: str = "RESOLVED_VALUE",
    value: float | None = 1000.0,
    source: str = "emdat",
    flagged: bool = False,
    flags: list[str] | None = None,
    run_type: str = "live",
    lower_bound: bool = False,
    provisional: bool = False,
    frozen_at: str | None = None,
    with_trigger: bool = True,
):
    """One ``haz_resolutions`` row (and, by default, its trigger row).

    ``with_trigger`` defaults on because a resolution without a trigger row
    is not a state the machine produces — every resolution follows a
    detection verdict — and a test that seeded one would be measuring a
    denominator that cannot occur.
    """
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    year, month = (int(p) for p in ym.split("-"))
    if with_trigger:
        seed_trigger(
            con,
            iso3=iso3,
            ym=ym,
            hazard=hazard,
            triggered=status != "RESOLVED_ZERO",
            run_type=run_type,
        )
    provenance = {
        "source": source,
        "source_record_ids": [f"{source}-{iso3}-{ym}"],
        "source_urls": [f"https://example.test/{source}/{iso3}/{ym}"],
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "rule_fired": f"ladder:{source}",
        "value_is_lower_bound": lower_bound,
        "decision": {"flags": list(flags or ([] if not flagged else ["unspecified"]))},
    }
    if status == "RESOLVED_ZERO":
        provenance["evidence_of_absence"] = {
            "reliefweb": {"total_hits": 0, "silent": True},
            "gdacs": {"total_records": 12, "query": {"n_events_qualifying_global": 0}},
        }
    con.execute(
        """
        INSERT OR REPLACE INTO haz_resolutions
            (iso3, year, month, hazard, status, value, provenance_json,
             rule_fired, flagged, provisional, run_type, frozen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            iso3.upper(), year, month, hazard, status, value,
            json.dumps(provenance), f"ladder:{source}", flagged, provisional,
            run_type, frozen_at,
        ],
    )


def seed_candidate(
    con,
    *,
    iso3: str,
    ym: str,
    hazard: str = "FL",
    source: str = "ifrc_go",
    value: float = 1000.0,
    value_type: str = "affected",
):
    """One ``haz_impact_candidates`` row (the acceptance baseline reads these)."""
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    year, month = (int(p) for p in ym.split("-"))
    con.execute(
        """
        INSERT OR REPLACE INTO haz_impact_candidates
            (iso3, year, month, hazard, value, value_type, source, source_ref)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [iso3.upper(), year, month, hazard, value, value_type, source,
         f"{source}-{iso3}-{ym}"],
    )


def seed_base_rates(
    con,
    *,
    iso3: str,
    hazard: str,
    occurrence: dict[int, tuple[float, int]] | None = None,
    quantiles: tuple[float, float, float, float, float] | None = None,
    n_events: int = 0,
    window: str = "2010-01..2026-05",
    window_start: int = 2015,
    window_end: int = 2026,
    provenance_mix: dict[str, Any] | None = None,
):
    """Seed published base rates for one country-hazard.

    ``occurrence`` maps calendar month -> (p_occurrence, n_years). A severity
    row is written only when ``quantiles`` is given, so a test can build the
    ASSESSED-BUT-NEVER-RESOLVED state (occurrence rows, no severity row) that
    the prompt block renders as "no historical events in record".
    """
    from resolver.hazard_resolution.schema import ensure_haz_schema

    ensure_haz_schema(con)
    for month, (probability, n_years) in sorted((occurrence or {}).items()):
        con.execute(
            """
            INSERT INTO haz_base_rates_occurrence
                (iso3, hazard, calendar_month, p_occurrence, n_years, source_window)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [iso3.upper(), hazard, int(month), float(probability), int(n_years), window],
        )
    if quantiles is not None:
        con.execute(
            """
            INSERT INTO haz_base_rates_severity
                (iso3, hazard, q10, q25, q50, q75, q90,
                 n_events, window_start, window_end, provenance_mix_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                iso3.upper(), hazard, *[float(q) for q in quantiles],
                int(n_events), int(window_start), int(window_end),
                json.dumps(provenance_mix or {}),
            ],
        )
