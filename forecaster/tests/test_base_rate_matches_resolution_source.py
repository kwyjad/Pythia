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
