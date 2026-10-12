# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A PA base rate must never be drawn from a wider series than the resolver.

The forecaster shows a model a historical base rate; ``compute_resolutions``
later scores that model against ground truth. If the base rate is built over
metrics the resolver will not use, the model is anchored on a series it will
never be scored against.

That is not hypothetical. ``_build_natural_hazard_seasonal_profile`` matched
``('affected', 'in_need', 'pa')`` while the resolver matched
``('affected', 'people_affected', 'pa', 'displaced')``. GDACS writes
``in_need`` for FL/DR/TC — modelled population exposure (hazard footprint x
population), which is orders of magnitude larger than an IFRC reported
"affected" figure and a different quantity in kind. So GDACS exposure entered
the seasonal profile presented to the model as IFRC people-affected history,
on exactly the hazards where IFRC coverage is weakest.

See docs/montandon_assessment.md.
"""

from __future__ import annotations

from datetime import date

import re

import duckdb
import pytest

from forecaster.history_loaders import _pa_metric_in_clause
from pythia.tools.compute_resolutions import (
    PA_FACTS_DELTAS_METRICS,
    PA_FACTS_RESOLVED_METRICS,
)


def _metrics_in_clause(clause: str) -> set[str]:
    """Pull the quoted metric literals back out of a SQL ``IN (...)`` clause."""
    return set(re.findall(r"'([^']+)'", clause))


def test_base_rate_metrics_are_a_subset_of_what_the_resolver_scores():
    base_rate_metrics = _metrics_in_clause(_pa_metric_in_clause())
    resolver_metrics = set(PA_FACTS_RESOLVED_METRICS)

    extra = base_rate_metrics - resolver_metrics
    assert not extra, (
        f"PA base rate is built over metrics the resolver will never score: "
        f"{sorted(extra)}. A model anchored on these is scored against a "
        f"different series. Resolver uses: {sorted(resolver_metrics)}."
    )


def test_in_need_is_never_a_pa_metric():
    """``in_need`` is modelled exposure, not reported impact — never in PA."""
    assert "in_need" not in PA_FACTS_RESOLVED_METRICS
    assert "in_need" not in PA_FACTS_DELTAS_METRICS
    assert "in_need" not in _metrics_in_clause(_pa_metric_in_clause())


def test_deltas_metrics_extend_resolved_metrics_only_with_new_displacements():
    """facts_deltas adds IDMC's flow metric and nothing else."""
    assert set(PA_FACTS_DELTAS_METRICS) - set(PA_FACTS_RESOLVED_METRICS) == {
        "new_displacements"
    }


@pytest.fixture()
def facts_db(tmp_path):
    """A facts_resolved table holding one IFRC row and one GDACS exposure row."""
    db = tmp_path / "facts.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT, hazard_code TEXT, ym TEXT, metric TEXT,
            value DOUBLE, publisher TEXT, source_id TEXT
        )
        """
    )
    con.executemany(
        "INSERT INTO facts_resolved VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            # What IFRC actually reported.
            ("PHL", "TC", "2025-03", "affected", 42_000.0, "IFRC", "ifrc_go"),
            # GDACS modelled exposure for the same country-hazard-month: three
            # orders of magnitude larger, and a different quantity entirely.
            ("PHL", "TC", "2025-03", "in_need", 8_400_000.0, "GDACS / JRC", "gdacs"),
        ],
    )
    con.close()
    return db


def test_gdacs_exposure_cannot_enter_a_pa_base_rate(facts_db):
    """The clause the base rate uses must not select the GDACS row."""
    con = duckdb.connect(str(facts_db), read_only=True)
    try:
        rows = con.execute(
            f"SELECT metric, value FROM facts_resolved "
            f"WHERE iso3 = 'PHL' AND hazard_code = 'TC' AND {_pa_metric_in_clause()}"
        ).fetchall()
    finally:
        con.close()

    assert rows == [("affected", 42_000.0)], (
        "The PA base-rate clause selected a GDACS 'in_need' exposure row. "
        "That figure is population inside a hazard footprint, not reported "
        "impact, and the resolver will never use it to score the question."
    )


def test_seasonal_profile_excludes_exposure_and_labels_its_real_source(
    facts_db, monkeypatch
):
    """End-to-end over the builder that actually runs in production.

    ``_build_natural_hazard_seasonal_profile`` is what ``_build_history_summary``
    calls for natural-hazard PA. Before the fix it reported the 8.4M GDACS
    exposure figure as the March maximum, labelled "IFRC".
    """
    import forecaster.cli as cli

    con = duckdb.connect(str(facts_db), read_only=True)
    monkeypatch.setattr(cli, "connect", lambda read_only=False: con)
    try:
        profile = cli._build_natural_hazard_seasonal_profile("PHL", "TC")
    finally:
        con.close()

    march = profile["months"][3]
    assert march["n_observations"] == 1, (
        "Expected only the IFRC reported figure; the GDACS exposure row "
        "leaked back into the seasonal profile."
    )
    assert march["max"] == 42_000
    assert profile["source"] == "IFRC"


def test_seasonal_profile_source_label_names_every_publisher_present(
    tmp_path, monkeypatch
):
    """A blended base rate must say so rather than asserting "IFRC"."""
    import forecaster.cli as cli

    db = tmp_path / "mixed.duckdb"
    setup = duckdb.connect(str(db))
    setup.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT, hazard_code TEXT, ym TEXT, metric TEXT,
            value DOUBLE, publisher TEXT, source_id TEXT
        )
        """
    )
    setup.executemany(
        "INSERT INTO facts_resolved VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("SOM", "FL", "2025-04", "affected", 1_000.0, "IFRC", "ifrc_go"),
            ("SOM", "FL", "2025-05", "displaced", 2_000.0, "IDMC", "idmc"),
        ],
    )
    setup.close()

    con = duckdb.connect(str(db), read_only=True)
    monkeypatch.setattr(cli, "connect", lambda read_only=False: con)
    try:
        profile = cli._build_natural_hazard_seasonal_profile("SOM", "FL")
    finally:
        con.close()

    assert profile["source"] == "IDMC, IFRC"


# ---------------------------------------------------------------------------
# ACE/PA: one conflict displacement series, one reader (Oct 2026)
# ---------------------------------------------------------------------------


def _conflict_db():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "series_semantics TEXT, value DOUBLE, publisher TEXT, source_id TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    rows = [
        ("2026-05", "PHL", "ACE", 12_000), ("2026-07", "PHL", "ACE", 3_000),
        ("2026-06", "SOM", "ACE", 900),
        # All-cause rows as they stood on the 3 October 2026 release: typhoon
        # evacuations, never conflict displacement.
        ("2026-07", "CHN", "IDU", 7_305_385), ("2026-09", "PHL", "IDU", 1_743_994),
    ]
    for ym, iso, hz, v in rows:
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, "
            "value, publisher, source_id) VALUES (?, ?, ?, 'new_displacements', 'new', ?, "
            "'IDMC', 'idmc')",
            [ym, iso, hz, v],
        )
    return con


def test_prompt_anchor_references_and_resolver_read_one_conflict_series():
    """The prompt block, the ACE/PA anchor, the persistence reference and the
    resolver all read IDMC CONFLICT displacement through one reader, and the
    writer names the same hazard, metric and publisher."""
    from pythia.tools import base_rate_spd as brs
    from resolver.ingestion import idmc_conflict as ic

    con = _conflict_db()
    # Prompt block reader.
    rows, _neg = brs.conflict_displacement_rows(con, "PHL", "2026-10", limit=6)
    assert rows == [("2026-05", 12000.0), ("2026-07", 3000.0)]
    # Anchor: the same rows, read as the resolver reads them on the day a
    # forecast for a December window was made (13 November): May and July
    # are settled; PHL reports too rarely for a missing month to be zero.
    probs, source, detail = brs.base_rate_spd(con, "PHL", "ACE", "PA", "2026-12")
    assert source.startswith("idmc_conflict")
    assert 1_743_994.0 not in detail["values"]
    assert detail["values"] == [12000.0, 3000.0]
    # Resolver, once the months have settled: a reported month resolves, a
    # missing month of an irregular reporter stays unknown, never zero.
    later = date(2027, 6, 1)
    assert brs.conflict_displacement_value(con, "PHL", "2026-07", today=later)[0] == 3000.0
    assert brs.conflict_displacement_value(con, "PHL", "2026-06", today=later) is None
    # ...and before it has settled, even a reported month waits.
    assert brs.conflict_displacement_value(con, "PHL", "2026-07", today=date(2026, 9, 1)) is None
    # China's typhoon rows put it in no conflict series at all.
    assert brs.conflict_displacement_value(con, "CHN", "2026-07", today=later) is None
    probs_chn, source_chn, _ = brs.base_rate_spd(con, "CHN", "ACE", "PA", "2026-12")
    assert probs_chn == [] and source_chn == brs.NO_BASE_RATE_SOURCE
    # Persistence: the newest month settled when the forecast was made.
    assert brs.last_observed_value(con, "PHL", "ACE", "PA", "2026-12")[:2] == (3000.0, "2026-07")
    # Writer and reader agree on what the series is.
    assert (ic.HAZARD_CODE, ic.METRIC, ic.SOURCE) == (
        brs.CONFLICT_DISPLACEMENT_HAZARD,
        brs.CONFLICT_DISPLACEMENT_METRIC,
        brs.CONFLICT_DISPLACEMENT_PUBLISHER,
    )


def _add_lower_rungs(con):
    """Rung 2 (an extracted OCHA/UN figure) and rung 3 (a DTM lower bound),
    in the tables the Part A sources write."""
    import json as _json

    from resolver.ingestion import conflict_displacement_extract as cde
    from resolver.ingestion import dtm_conflict as dtm

    cde.ensure_schema(con)
    dtm.ensure_schema(con)
    con.execute(
        "INSERT INTO conflict_displacement_candidates (iso3, ym, source_ref, preference_rank, "
        "value_type, value_persons, quote, doc_url) VALUES "
        "('PHL', '2026-06', 'rw:1', 0, 'new_displacements', 8000, "
        "'some 8,000 people were newly displaced', 'https://reliefweb.int/report/1')"
    )
    con.execute(
        "INSERT INTO dtm_conflict_lower_bounds (iso3, ym, value, value_type, operation, "
        "n_pairs, pairs_json, operations_json, max_gap_days) VALUES "
        "('PHL', '2026-08', 1500, 'displaced_lower_bound', 'op', 1, ?, '{}', 35)",
        [_json.dumps([{"from": "2026-07-20", "to": "2026-08-15", "gap_days": 26}])],
    )


def test_prompt_anchor_references_and_resolver_read_one_conflict_ladder():
    """The ladder (IDMC, then an extracted figure, then a DTM lower bound) is
    read by the same readers: the resolver, the anchor, the persistence
    reference, the prompt rows, the scoring class and eligibility."""
    from pythia.tools import base_rate_spd as brs

    con = _conflict_db()
    _add_lower_rungs(con)
    later = date(2027, 6, 1)
    # Resolver: IDMC's months, then rung 2 for June, rung 3 for August.
    assert brs.conflict_displacement_value(con, "PHL", "2026-07", today=later) == (
        3000.0, brs.CONFLICT_DISPLACEMENT_SERIES)
    june = brs.conflict_displacement_value(con, "PHL", "2026-06", today=later)
    assert june[0] == 8000.0 and "reliefweb_extracted" in june[1]
    aug = brs.conflict_displacement_value(con, "PHL", "2026-08", today=later)
    assert aug[0] == 1500.0 and "lower_bound" in aug[1]
    # Anchor at the forecast date for a December window (13 Nov): May, June
    # (extracted, available 30 days after June), July, August (DTM round of
    # 15 Aug) are all known; the window reads four values.
    _probs, source, detail = brs.base_rate_spd(con, "PHL", "ACE", "PA", "2026-12")
    assert sorted(detail["values"]) == sorted([12000.0, 8000.0, 3000.0, 1500.0])
    # Persistence: the newest month resolved on the forecast day is August, DTM.
    val, ym, src = brs.last_observed_value(con, "PHL", "ACE", "PA", "2026-12")
    assert (val, ym) == (1500.0, "2026-08") and "lower_bound" in src
    # Prompt rows carry the rung per month and mark the lower bound.
    rows = brs.conflict_displacement_settled_rows(con, "PHL", later)
    by_ym = {r["ym"]: r for r in rows["rows_detail"]}
    assert by_ym["2026-06"]["rung"] == brs.CONFLICT_RUNG_EXTRACTED
    assert by_ym["2026-08"]["lower_bound"] is True
    # Eligibility and the scoring class count any rung.
    ladder = brs.conflict_ladder_reported(con, before_ym="2026-09")
    assert set(ladder["PHL"]) == {"2026-05", "2026-06", "2026-07", "2026-08"}


def test_the_ace_pa_prompt_names_the_rung_per_month():
    from forecaster.history_loaders import _conflict_ladder_lines

    lines = _conflict_ladder_lines([
        {"ym": "2026-06", "value": 8000.0, "rung": "reliefweb_extracted", "lower_bound": False},
        {"ym": "2026-08", "value": 1500.0, "rung": "dtm_lower_bound", "lower_bound": True},
    ])
    text = "\n".join(lines)
    assert "IDMC, then OCHA/UN report (ReliefWeb), then IOM DTM increase, a lower bound" in text
    assert "2026-08 at least 1,500 (IOM DTM increase, a lower bound)" in text
    # IDMC alone: the resolution line, no per-month list.
    only = _conflict_ladder_lines([{"ym": "2026-06", "value": 10.0, "rung": "idmc"}])
    assert len(only) == 1
