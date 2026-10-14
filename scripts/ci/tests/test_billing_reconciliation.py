# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Billed against recorded (Oct 2026). The providers are stubbed; their
response shapes follow the providers' documentation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci import billing_reconciliation as br  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "b.duckdb"))
    con.execute("CREATE TABLE llm_calls (provider TEXT, cost_usd DOUBLE, timestamp TIMESTAMP)")
    con.execute("""INSERT INTO llm_calls VALUES
        ('anthropic', 60.0, TIMESTAMP '2026-09-13 05:00:00'),
        ('anthropic', 40.0, TIMESTAMP '2026-09-30 23:59:00'),
        ('openai', 10.0, TIMESTAMP '2026-09-02 00:00:00'),
        ('google', 5.0, TIMESTAMP '2026-09-02 00:00:00'),
        ('anthropic', 999.0, TIMESTAMP '2026-10-01 00:00:00')""")
    return con


def _fake_get(url, headers, params):
    if "anthropic" in url:
        if params.get("page") is None:
            return {"data": [{"results": [{"amount": "6000"}]}], "has_more": True, "next_page": "p2"}
        return {"data": [{"results": [{"amount": "4100"}]}], "has_more": False}
    return {"data": [{"results": [{"amount": {"value": 13.0, "currency": "usd"}}]}], "has_more": False}


def test_previous_month_and_bounds():
    assert br.previous_month(date(2026, 11, 1)) == "2026-10"
    assert br.previous_month(date(2026, 1, 1)) == "2025-12"
    start, end = br.month_bounds("2026-12")
    assert (start.month, end.year, end.month) == (12, 2027, 1)


def test_a_month_is_reconciled_per_provider(tmp_path):
    con = _db(tmp_path)
    rows = br.reconcile(con, "2026-09", get=_fake_get,
                        env={"ANTHROPIC_ADMIN_API_KEY": "a", "OPENAI_ADMIN_API_KEY": "o"})
    by = {r["provider"]: r for r in rows}
    # Anthropic billed 101.00 (cents across two pages); the ledger holds 100.00.
    assert by["anthropic"]["billed_usd"] == pytest.approx(101.0)
    assert by["anthropic"]["status"] == "ok"
    # OpenAI billed 13.00 against 10.00 recorded: 23% low, a mismatch.
    assert by["openai"]["status"] == "mismatch" and by["openai"]["diff_pct"] == pytest.approx(-23.1)
    assert by["google"]["status"] == "unavailable" and "not available" in by["google"]["reason"]
    br.store(con, rows)
    br.store(con, rows)  # a rerun rewrites the month
    assert con.execute("SELECT COUNT(*) FROM billing_reconciliation").fetchone()[0] == 3


def test_no_admin_key_says_so(tmp_path):
    rows = br.reconcile(_db(tmp_path), "2026-09", get=_fake_get, env={})
    assert all(r["status"] == "unavailable" for r in rows)
    assert "ANTHROPIC_ADMIN_API_KEY not set" in rows[0]["reason"]


def test_main_never_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_API_KEY", raising=False)
    _db(tmp_path).close()
    out = tmp_path / "r.json"
    assert br.main(["--db", str(tmp_path / "b.duckdb"), "--month", "2026-09", "--out", str(out)]) == 0
    assert out.exists()
    assert br.main(["--db", str(tmp_path / "missing" / "x.duckdb"), "--out", str(out)]) == 0


def test_the_workflow_is_a_guarded_canonical_producer():
    wf = (REPO / ".github/workflows/billing_reconciliation.yml").read_text()
    assert "group: pythia-resolver-db" in wf and "guard-canonical-upload" in wf
    assert "extra-workflows: Billing Reconciliation" in wf
    for path in (".github/actions/download-canonical-db/action.yml",
                 ".github/workflows/publish_latest_data.yml",
                 ".github/workflows/run_horizon_scanner.yml"):
        assert "Billing Reconciliation" in (REPO / path).read_text(), path
