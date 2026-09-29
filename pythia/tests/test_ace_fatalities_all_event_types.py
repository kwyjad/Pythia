# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACE/FATALITIES resolves from the series its base rate uses (Sept 2026).

The August 2026 scored run resolved 27 of 32 ACE/FATALITIES questions to the
ACLED BATTLES-ONLY series: the connector writes ``fatalities_battle_month``,
the ACLED adapter renamed it ``fatalities``, and ``compute_resolutions`` read
``facts_resolved`` 'fatalities' before ``acled_monthly_fatalities`` — the
all-event-types series the question wording, the prompt base rate and the
climatology reference (``base_rate_spd``) are all defined on.

These tests pin every part of the repair: the adapter no longer renames, the
resolver reads only the all-types series and names it, the repair pass
relabels rows already stored, the centroid reset script, and the advice
generator's guards (reference forecasters excluded, one run per question,
blocked groups, no cross-hazard fallback).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pythia.tools import base_rate_spd as brs
from pythia.tools import compute_resolutions as cr
from resolver.tools import repair_acled_fatalities_metric as relabel
from resolver.transform.adapters import acled as acled_adapter


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _staging_rows() -> pd.DataFrame:
    """Two ACLED connector rows in the staging (21-column) shape."""

    base = {
        "country_name": "Brazil", "iso3": "BRA", "hazard_code": "ACE",
        "hazard_label": "Armed Conflict Escalation", "hazard_class": "conflict",
        "series_semantics": "new", "unit": "persons",
        "publication_date": "2026-09-10", "publisher": "ACLED",
        "source_type": "other", "source_url": "https://acleddata.com",
        "doc_title": "ACLED monthly aggregation", "definition_text": "Battle fatalities",
        "method": "api", "confidence": "", "revision": "0", "ingested_at": "2026-09-10",
    }
    rows = [
        {**base, "event_id": "BRA-ACLED-ACE-fatalities_battle_month-2026-08-a",
         "metric": "fatalities_battle_month", "value": "37", "as_of_date": "2026-08"},
        {**base, "event_id": "BRA-ACLED-ACE-events-2026-08-b",
         "metric": "events", "value": "12", "as_of_date": "2026-08"},
    ]
    return pd.DataFrame(rows)


def _facts_schema(con) -> None:
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT NOT NULL, iso3 TEXT NOT NULL, hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL, series_semantics TEXT NOT NULL DEFAULT '',
            value DOUBLE, publisher TEXT, source_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP,
            CONSTRAINT facts_resolved_unique UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT NOT NULL, iso3 TEXT NOT NULL, hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL, value_new DOUBLE, source_id TEXT,
            series_semantics TEXT NOT NULL DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP,
            CONSTRAINT facts_deltas_unique UNIQUE (ym, iso3, hazard_code, metric)
        )
        """
    )


def _store_adapter_output(con, frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, value, publisher) "
            "VALUES (?, ?, ?, ?, ?, ?, 'ACLED')",
            [row.as_of_date[:7], row.iso3, row.hazard_code, row.metric,
             row.series_semantics, float(row.value)],
        )


def _e2e_db(path: Path):
    con = duckdb.connect(str(path))
    _facts_schema(con)
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO hs_runs VALUES ('run1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, status TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    return con


# ---------------------------------------------------------------------------
# the adapter and the stored rows
# ---------------------------------------------------------------------------


def test_adapter_output_never_reaches_the_resolution_series_name() -> None:
    """FAILS against the pre-fix mapping: the battles-only series must keep
    its own name, so no ACLED row can be named 'fatalities'."""

    adapter = acled_adapter.ACLEDAdapter("acled")
    frame = adapter.map(_staging_rows())
    con = duckdb.connect(":memory:")
    _facts_schema(con)
    _store_adapter_output(con, frame)
    assert relabel.count_mislabelled(con) == {"facts_resolved": 0, "facts_deltas": 0}
    assert "fatalities_battle_month" in set(frame["metric"])


def test_the_pre_fix_mapping_is_what_the_check_catches(monkeypatch) -> None:
    """The guard above is not vacuous: with the old rename it finds the row."""

    monkeypatch.setattr(
        acled_adapter, "_METRIC_MAP", {"fatalities_battle_month": "fatalities"}
    )
    frame = acled_adapter.ACLEDAdapter("acled").map(_staging_rows())
    con = duckdb.connect(":memory:")
    _facts_schema(con)
    _store_adapter_output(con, frame)
    assert relabel.count_mislabelled(con)["facts_resolved"] == 1


@pytest.mark.db
def test_resolution_series_equals_the_base_rate_series(tmp_path: Path, monkeypatch) -> None:
    """A battles-only facts row beside the all-types series: the question
    resolves to the all-types figure, and the series it names is the one
    base_rate_spd builds the climatology anchor from."""

    db = tmp_path / "e2e.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = _e2e_db(db)
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher) "
        "VALUES ('2025-08', 'BRA', 'ACE', 'fatalities', 37, 'ACLED')"
    )
    for ym, n in [("2025-06-01", 400), ("2025-07-01", 410), ("2025-08-01", 412)]:
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES ('BRA', ?, ?, 'ACLED', "
            "TIMESTAMP '2025-09-10 00:00:00')",
            [ym, n],
        )
    con.execute(
        "INSERT INTO questions VALUES ('BRA_ACE_FATALITIES_2025-08', 'run1', 'BRA', 'ACE', "
        "'FATALITIES', '2026-01', DATE '2025-08-01', 'active', FALSE)"
    )
    # A stale resolution from the battles-only facts row, as August wrote it.
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, observed_month TEXT, "
        "value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT, created_at TIMESTAMP, "
        "is_test BOOLEAN DEFAULT FALSE, PRIMARY KEY (question_id, horizon_m))"
    )
    con.execute(
        "INSERT INTO resolutions VALUES ('BRA_ACE_FATALITIES_2025-08', 1, '2025-08', 37, NULL, "
        "'facts_resolved:ACLED:fatalities', now(), FALSE)"
    )
    con.close()

    cr.compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db))
    rows = con.execute(
        "SELECT horizon_m, value, source_desc FROM resolutions ORDER BY horizon_m"
    ).fetchall()
    probs, base_source, _detail = brs.base_rate_spd(con, "BRA", "ACE", "FATALITIES", "2025-08")
    con.close()

    assert rows == [(1, 412.0, cr.ACE_FATALITIES_SERIES)]
    assert probs, "base_rate_spd found no anchor for the fixture"
    assert base_source.split(":")[0] == cr.ACE_FATALITIES_TABLE == brs.CONFLICT_FATALITIES_TABLE
    assert rows[0][2].split(":")[0] == base_source.split(":")[0]


def test_freshness_and_coverage_ignore_facts_fatalities(tmp_path: Path) -> None:
    """A facts 'fatalities' row (IFRC deaths, legacy battles-only) must not
    extend the FATALITIES cutoff or the coverage gates."""

    con = _e2e_db(tmp_path / "c.duckdb")
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher) "
        "VALUES ('2026-08', 'PHL', 'TC', 'fatalities', 12, 'IFRC')"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('SOM', DATE '2026-06-01', 5, 'ACLED', now())"
    )
    assert cr._data_freshness_cutoff(con, "FATALITIES") == "2026-06"
    from pythia.tools.source_coverage import (
        countries_with_source_data,
        refresh_source_coverage,
    )

    refresh_source_coverage(con)
    assert countries_with_source_data(con, "FATALITIES") == {"SOM"}


# ---------------------------------------------------------------------------
# the repair pass
# ---------------------------------------------------------------------------


def test_repair_relabels_merges_and_is_idempotent() -> None:
    con = duckdb.connect(":memory:")
    _facts_schema(con)
    con.execute(
        """
        INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher) VALUES
          ('2026-07', 'MEX', 'ACE', 'fatalities', 80, 'ACLED'),
          ('2026-08', 'MEX', 'ACE', 'fatalities', 90, 'ACLED'),
          ('2026-08', 'MEX', 'ACE', 'fatalities_battle_month', 91, 'ACLED'),
          ('2026-08', 'PHL', 'TC', 'fatalities', 12, 'IFRC')
        """
    )
    con.execute(
        "INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, source_id) VALUES "
        "('2026-07', 'MEX', 'ACE', 'fatalities', 80, 'acled')"
    )

    report = relabel.repair(con)
    assert report["tables"]["facts_resolved"] == {"relabelled": 1, "merged": 1, "untouched": 2}
    assert report["tables"]["facts_deltas"]["relabelled"] == 1
    assert relabel.count_mislabelled(con) == {"facts_resolved": 0, "facts_deltas": 0}
    # The merge kept the correctly named row written after the fix.
    assert con.execute(
        "SELECT value FROM facts_resolved WHERE iso3='MEX' AND ym='2026-08'"
    ).fetchall() == [(91.0,)]
    # IFRC natural-hazard deaths are not ACLED's and are left alone.
    assert con.execute(
        "SELECT metric FROM facts_resolved WHERE iso3='PHL'"
    ).fetchall() == [("fatalities",)]

    again = relabel.repair(con)
    assert again["tables"]["facts_resolved"]["relabelled"] == 0
    assert again["tables"]["facts_resolved"]["merged"] == 0


# ---------------------------------------------------------------------------
# centroids
# ---------------------------------------------------------------------------


def test_reset_conflict_centroids_falls_back_to_seeds() -> None:
    from pythia.tools.compute_scores import _load_centroids
    from pythia.buckets import centroids_for, n_buckets_for
    from scripts import reset_conflict_centroids as reset_mod

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE bucket_centroids (hazard_code TEXT, metric TEXT, bucket_index INTEGER, "
        "centroid DOUBLE, as_of_month TEXT, schema_version TEXT)"
    )
    k = n_buckets_for("FATALITIES")
    seeds = list(centroids_for("FATALITIES"))
    for i in range(1, k + 1):
        con.execute(
            "INSERT INTO bucket_centroids VALUES ('*', 'FATALITIES', ?, ?, NULL, NULL)",
            [i, seeds[i - 1]],
        )
        con.execute(
            "INSERT INTO bucket_centroids VALUES ('ACE', 'FATALITIES', ?, ?, '2026-09', NULL)",
            [i, seeds[i - 1] * 0.4],
        )
        con.execute(
            "INSERT INTO bucket_centroids VALUES ('FL', 'PA', ?, 1.0, '2026-09', NULL)", [i]
        )

    report = reset_mod.reset(con)
    assert report["deleted"] == k
    assert _load_centroids(con, "ACE", "FATALITIES", k) == pytest.approx(seeds)
    assert con.execute(
        "SELECT COUNT(*) FROM bucket_centroids WHERE hazard_code = 'FL'"
    ).fetchone()[0] == k
    assert reset_mod.reset(con)["deleted"] == 0


def test_audit_export_is_read_only_and_degrades(tmp_path: Path) -> None:
    from scripts import reset_conflict_centroids as reset_mod

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT)")
    con.execute("INSERT INTO questions VALUES ('Q1','ACE','FATALITIES'), ('Q2','FL','PA')")
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    con.execute("INSERT INTO resolutions VALUES ('Q1',1,37), ('Q2',1,5)")
    counts = reset_mod.export_audit(con, tmp_path / "audit")
    assert counts["resolutions"] == 1
    assert counts["scores"] == -1  # absent table: recorded, not raised
    text = (tmp_path / "audit" / "resolutions__ace_fatalities.csv").read_text()
    assert "Q1" in text and "Q2" not in text
    assert con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# calibration advice
# ---------------------------------------------------------------------------


def _advice_db():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.execute(
        "CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, horizon_m INTEGER, "
        "class_bin TEXT, p DOUBLE)"
    )
    con.execute("INSERT INTO questions VALUES ('Q1','ACE','FATALITIES',FALSE)")
    con.execute(
        "INSERT INTO forecasts_ensemble VALUES ('Q1','fc_100',1,'0',1.0), ('Q1','fc_200',1,'0',1.0)"
    )
    con.execute(
        """
        INSERT INTO scores VALUES
          ('Q1',1,'brier','model-a',0.9,'fc_100'),
          ('Q1',1,'brier','model-a',0.2,'fc_200'),
          ('Q1',1,'brier','model-b',0.4,'fc_200'),
          ('Q1',1,'brier','ensemble_mean_v2',0.3,'fc_200'),
          ('Q1',1,'brier','__ext_climatology',0.05,NULL),
          ('Q1',1,'brier','__ext_uniform',0.8,NULL)
        """
    )
    return con


def test_per_model_brier_excludes_reference_forecasters_and_old_runs() -> None:
    from pythia.tools.generate_calibration_advice import _compute_per_model_brier

    out = _compute_per_model_brier(_advice_db(), "ACE", "FATALITIES")
    names = {m["name"]: m for m in out["all_models"]}
    assert set(names) == {"model-a", "model-b"}
    # Only the latest run (fc_200) counts: 0.2, not the mean with fc_100's 0.9.
    assert names["model-a"]["brier"] == pytest.approx(0.2)
    assert names["model-a"]["n"] == 1
    assert out["best"]["name"] == "model-a"


def test_blocked_groups_parse_and_withhold_prompt_advice(tmp_path: Path, monkeypatch) -> None:
    from forecaster import prompts
    from pythia.tools.generate_calibration_advice import advice_blocked_groups

    monkeypatch.setenv("PYTHIA_ADVICE_BLOCK_GROUPS", " ace/fatalities , FL/PA,junk")
    assert advice_blocked_groups() == {("ACE", "FATALITIES"), ("FL", "PA")}

    db = tmp_path / "adv.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, advice_version TEXT)"
    )
    con.execute(
        "INSERT INTO calibration_advice VALUES "
        "('2026-09','ACE','FATALITIES','__shared__','CONFLICT ADVICE','v1'), "
        "('2026-09','DR','PHASE3PLUS_IN_NEED','__shared__','DROUGHT ADVICE','v1')"
    )
    con.close()
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: f"duckdb:///{db}")

    assert prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES") == ""
    monkeypatch.setenv("PYTHIA_ADVICE_BLOCK_GROUPS", "")
    assert prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES") == "CONFLICT ADVICE"
    # No cross-hazard fallback: a flood question with no advice of its own
    # (and no global row) gets nothing, never the drought or conflict text.
    assert prompts._load_calibration_advice_for_hazard("FL", "PA") == ""


def test_generation_skips_a_group_whose_resolutions_predate_the_fix(tmp_path: Path) -> None:
    from pythia.tools import generate_calibration_advice as gca

    db = tmp_path / "gen.duckdb"
    con = _advice_db()
    con.execute(f"ATTACH '{db}' AS out")
    for table in ("questions", "scores", "forecasts_ensemble"):
        con.execute(f"CREATE TABLE out.{table} AS SELECT * FROM {table}")
    con.execute(
        "CREATE TABLE out.resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE, "
        "observed_month TEXT, source_desc TEXT)"
    )
    con.execute(
        "INSERT INTO out.resolutions VALUES ('Q1', 1, 37, '2026-08', 'facts_resolved:ACLED:fatalities')"
    )
    con.execute(
        "CREATE TABLE out.calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, findings_json TEXT, advice_version TEXT, created_at TIMESTAMP, "
        "PRIMARY KEY (as_of_month, hazard_code, metric, model_name))"
    )
    con.execute(
        "INSERT INTO out.calibration_advice VALUES "
        "('2026-09','ACE','FATALITIES','__shared__','learned from battles',NULL,'v1',now()), "
        "('2026-09','ACE','FATALITIES','__ext_climatology','advice to a reference',NULL,'v1',now())"
    )
    con.close()

    assert gca._fatalities_resolutions_predate_fix(duckdb.connect(str(db)), "ACE", "FATALITIES") == 1

    from datetime import date

    gca.generate_calibration_advice(f"duckdb:///{db}", as_of=date(2026, 9, 29))
    con = duckdb.connect(str(db))
    left = con.execute("SELECT model_name FROM calibration_advice").fetchall()
    con.close()
    assert left == []
