# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The debug bundle must say when a run lost the batch discount.

Before this, the bundle contained ZERO references to llm_batches or
service_tier. For the 2026-07-29 reference run it reported "LLM Calls: OK, 80
calls, 0 errors" — true, and useless: that run had lost the discount on two
thirds of its spend and had two OpenAI batches return nothing at all. The
bundle is the artifact a human reads first, and it said everything was fine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import _load_batch_health

HS_RUN = "hs_20260729T071137"
FC_RUN = "fc_1785315556"
PRED = "(hs_run_id = ? OR run_id = ?)"
PARAMS = [HS_RUN, FC_RUN]


def _db(tmp_path: Path, *, with_batches: bool = True):
    con = duckdb.connect(str(tmp_path / "bundle_batch.duckdb"))
    if with_batches:
        con.execute(
            """
            CREATE TABLE llm_batches (
                batch_id TEXT, provider TEXT, family TEXT, status TEXT,
                hs_run_id TEXT, run_id TEXT, n_requests INTEGER,
                n_succeeded INTEGER, error_text TEXT
            )
            """
        )
        con.execute(
            "CREATE TABLE llm_batch_requests (custom_id TEXT, batch_id TEXT, status TEXT)"
        )
    con.execute(
        """
        CREATE TABLE llm_calls (
            hs_run_id TEXT, run_id TEXT, cost_usd DOUBLE, usage_json TEXT
        )
        """
    )
    return con


def _reference_run(con):
    """The real shape of the reference run: OpenAI batches empty, others fine."""
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('b_openai_spd','openai','spd_v2','collected',NULL,?,12,0,'')",
        [FC_RUN],
    )
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('b_google_spd','google','spd_v2','collected',NULL,?,34,34,'')",
        [FC_RUN],
    )
    for i in range(16):
        con.execute(
            "INSERT INTO llm_batch_requests VALUES (?, 'b_openai_spd', 'fallback_sync')",
            [f"f{i}"],
        )
    for i in range(34):
        con.execute(
            "INSERT INTO llm_batch_requests VALUES (?, 'b_google_spd', 'succeeded')",
            [f"s{i}"],
        )
    # The expensive members are the ones that lost the discount.
    con.execute("INSERT INTO llm_calls VALUES (NULL, ?, 2.24, '{}')", [FC_RUN])
    con.execute(
        "INSERT INTO llm_calls VALUES (NULL, ?, 1.0456, '{\"service_tier\": \"batch\"}')",
        [FC_RUN],
    )


def test_empty_batch_and_cost_share_are_surfaced(tmp_path):
    con = _db(tmp_path)
    _reference_run(con)
    bh = _load_batch_health(con, HS_RUN, FC_RUN, PRED, PARAMS)

    assert bh["available"] is True
    assert bh["fallback_pct"] == pytest.approx(32.0)
    # The cost share is the point: a request count understates it by 2x.
    assert bh["full_price_pct_of_spend"] > bh["fallback_pct"]
    assert [e["provider"] for e in bh["empty_batches"]] == ["openai"]


def test_healthy_run_reports_no_empty_batches(tmp_path):
    con = _db(tmp_path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('b_ok','google','spd_v2','collected',NULL,?,4,4,'')",
        [FC_RUN],
    )
    for i in range(4):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'b_ok', 'succeeded')", [f"s{i}"])
    con.execute(
        "INSERT INTO llm_calls VALUES (NULL, ?, 1.0, '{\"service_tier\": \"batch\"}')",
        [FC_RUN],
    )

    bh = _load_batch_health(con, HS_RUN, FC_RUN, PRED, PARAMS)
    assert bh["empty_batches"] == []
    assert bh["fallback_pct"] == pytest.approx(0.0)
    assert bh["full_price_pct_of_spend"] == pytest.approx(0.0)


def test_other_runs_in_the_travelling_db_are_excluded(tmp_path):
    """llm_batches accumulates across runs; a prior pipeline is not this one."""
    con = _db(tmp_path)
    _reference_run(con)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('b_old','openai','spd_v2','collected',NULL,'fc_previous',99,0,'')"
    )
    bh = _load_batch_health(con, HS_RUN, FC_RUN, PRED, PARAMS)
    assert bh["n_batches"] == 2
    assert all(e["batch_id"] != "b_old" for e in bh["empty_batches"])


def test_pre_batch_db_degrades_quietly(tmp_path):
    """A DB with no batch tables must return unavailable, never raise."""
    con = _db(tmp_path, with_batches=False)
    assert _load_batch_health(con, HS_RUN, FC_RUN, PRED, PARAMS) == {"available": False}


def test_no_run_ids_returns_unavailable(tmp_path):
    con = _db(tmp_path)
    _reference_run(con)
    assert _load_batch_health(con, None, None, PRED, PARAMS) == {"available": False}
