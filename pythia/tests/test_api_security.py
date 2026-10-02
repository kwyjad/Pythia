# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The public API's request-level protections.

Each case here is one thing a stranger could do to the API once the
dashboard is shared: read a token out of a URL, keep the instance busy,
learn the route map, or turn a query into a file read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
import pythia.api.core as _core
from pythia.api import security
from pythia.api.app import app

TOKEN = "debug-token-for-tests-0123456789"


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE questions (question_id TEXT, iso3 TEXT, is_test BOOLEAN)")
    con.execute("INSERT INTO questions VALUES ('q1', 'ETH', FALSE)")
    con.close()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("FRED_DEBUG_TOKEN", TOKEN)
    monkeypatch.delenv("PYTHIA_API_EXTERNAL_ACCESS", raising=False)
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    _app_mod._FORCE_SYNC_LAST_AT = None
    security.LIMITER.reset()
    try:
        yield db_path
    finally:
        if _core._READ_CON is not None:
            try:
                _core._READ_CON.close()
            except Exception:
                pass
        _app_mod._READ_CON = None
        _app_mod._FORCE_SYNC_LAST_AT = None
        security.LIMITER.reset()
        pythia_config.load.cache_clear()


# --- tokens -----------------------------------------------------------------


def test_force_sync_refuses_a_token_in_the_url(api_env):
    client = TestClient(app)
    resp = client.post("/v1/admin/force_sync", params={"token": TOKEN})
    assert resp.status_code == 400
    assert "header" in resp.json()["detail"]


def test_force_sync_refuses_a_wrong_header_token(api_env):
    client = TestClient(app)
    resp = client.post("/v1/admin/force_sync", headers={"X-Fred-Debug-Token": "wrong"})
    assert resp.status_code == 403


def test_force_sync_runs_once_then_cools_down(api_env, monkeypatch):
    monkeypatch.setattr(_app_mod, "maybe_sync_latest_db", lambda wait=True: {"db_sha256": "x"})
    monkeypatch.setattr(_app_mod, "db_was_refreshed", lambda: False)
    client = TestClient(app)
    headers = {"X-Fred-Debug-Token": TOKEN}
    first = client.post("/v1/admin/force_sync", headers=headers)
    assert first.status_code == 200, first.text
    second = client.post("/v1/admin/force_sync", headers=headers)
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0


def test_the_debug_token_comparison_is_constant_time(monkeypatch):
    calls = []
    real = _core.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setenv("FRED_DEBUG_TOKEN", TOKEN)
    monkeypatch.setattr(_core.hmac, "compare_digest", spy)
    _core._require_debug_token(TOKEN)
    assert calls, "the token was not compared with hmac.compare_digest"
    with pytest.raises(Exception):
        _core._require_debug_token(None)


def test_the_fail_open_token_dependency_is_gone():
    from pythia.api import auth

    assert not hasattr(auth, "require_token")


# --- disclosure -------------------------------------------------------------


def test_memory_diagnostics_need_the_debug_token(api_env):
    client = TestClient(app)
    assert client.get("/v1/diagnostics/memory").status_code == 403
    ok = client.get("/v1/diagnostics/memory", headers={"X-Fred-Debug-Token": TOKEN})
    assert ok.status_code == 200
    assert "rss_mb" in ok.json()


def test_docs_and_schema_are_off_by_default():
    client = TestClient(app)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_responses_carry_security_headers(api_env):
    client = TestClient(app)
    resp = client.get("/v1/health")
    for name, value in security.SECURITY_HEADERS.items():
        assert resp.headers.get(name) == value, name


# --- load -------------------------------------------------------------------


def test_the_rate_limiter_answers_429_once_a_bucket_is_spent(api_env, monkeypatch):
    monkeypatch.setenv("PYTHIA_RATE_LIMIT_SCALE", "0.5")
    client = TestClient(app)
    bucket = security.bucket_for("/v1/diagnostics/memory")
    capacity = int(bucket.capacity * 0.5)
    headers = {"X-Fred-Debug-Token": TOKEN, "X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
    codes = [client.get("/v1/diagnostics/memory", headers=headers).status_code for _ in range(capacity + 2)]
    assert codes[:capacity] == [200] * capacity
    assert codes[-1] == 429
    refused = client.get("/v1/diagnostics/memory", headers=headers)
    assert refused.headers.get("X-Content-Type-Options") == "nosniff"
    # Another client, keyed on its own first forwarded hop, is unaffected.
    other = {"X-Fred-Debug-Token": TOKEN, "X-Forwarded-For": "198.51.100.9"}
    assert client.get("/v1/diagnostics/memory", headers=other).status_code == 200


def test_health_is_never_rate_limited(api_env, monkeypatch):
    monkeypatch.setenv("PYTHIA_RATE_LIMIT_SCALE", "0.001")
    client = TestClient(app)
    assert all(client.get("/v1/health").status_code == 200 for _ in range(20))


def test_buckets_refill_with_time():
    clock = {"t": 0.0}
    limiter = security.RateLimiter(clock=lambda: clock["t"])
    bucket = security.Bucket("t", capacity=2, per_minute=60)
    assert limiter.take("c", bucket) is None
    assert limiter.take("c", bucket) is None
    assert limiter.take("c", bucket) is not None
    clock["t"] = 1.0
    assert limiter.take("c", bucket) is None


def test_downloads_refuse_a_free_text_hazard_or_model(api_env):
    client = TestClient(app)
    assert client.get("/v1/downloads/rationales.csv", params={"hazard": "../../x"}).status_code == 400
    bad_model = {"hazard": "FL", "model": "a b;c"}
    assert client.get("/v1/downloads/rationales.csv", params=bad_model).status_code == 400


# --- the serving connection -------------------------------------------------


def test_the_serving_connection_cannot_read_other_files(api_env):
    con = _core._open_duckdb_connection()
    try:
        assert con.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('/etc/hostname')").fetchall()
        with pytest.raises(duckdb.Error):
            con.execute("SET enable_external_access=true")
    finally:
        con.close()
