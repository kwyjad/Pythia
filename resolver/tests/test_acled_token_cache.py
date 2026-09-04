# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One ACLED token per run, and the password grant before the refresh grant.

Run 33841370196 requested a token separately for acled_client, acled_cast
and pythia.acled_political (plus retries), and each request first tried a
refresh token that had expired weeks earlier — seven ERROR lines whose
normal outcome was failure. Tokens live 86,400 seconds; one is enough.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from resolver.ingestion import acled_auth


def _jwt(exp: int) -> str:
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{b64({'alg': 'none'})}.{b64({'exp': exp})}.sig"


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    path = tmp_path / "token.json"
    monkeypatch.setenv("ACLED_TOKEN_CACHE_PATH", str(path))
    for name in ("ACLED_ACCESS_TOKEN", "ACLED_TOKEN", "ACLED_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ACLED_USERNAME", "user@example.org")
    monkeypatch.setenv("ACLED_PASSWORD", "pw")
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})
    yield path
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})


def test_a_second_process_reuses_the_token_the_first_obtained(cache, monkeypatch):
    calls = []
    token = _jwt(int(time.time()) + 86_400)

    def fake_password(username, password):
        calls.append("password")
        return {"access_token": token, "refresh_token": "r1"}

    monkeypatch.setattr(acled_auth, "_password_grant", fake_password)
    assert acled_auth.get_access_token() == token
    assert calls == ["password"]
    assert cache.is_file()
    saved = json.loads(cache.read_text())
    assert saved["access_token"] == token and saved["refresh_token"] == "r1"

    # "Another process": the in-memory cache is empty, the file is not.
    acled_auth._CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})
    assert acled_auth.get_access_token() == token
    assert calls == ["password"]           # no second grant


def test_a_cached_token_near_expiry_is_not_reused(cache, monkeypatch):
    stale = _jwt(int(time.time()) + 60)    # inside _MIN_TTL
    cache.write_text(json.dumps({
        "access_token": stale, "refresh_token": None,
        "expiry": int(time.time()) + 60,
        "fingerprint": acled_auth._credential_fingerprint(),
    }))
    fresh = _jwt(int(time.time()) + 86_400)
    monkeypatch.setattr(acled_auth, "_password_grant", lambda u, p: {"access_token": fresh})
    assert acled_auth.get_access_token() == fresh


def test_a_cache_written_for_another_account_is_ignored(cache, monkeypatch):
    other = _jwt(int(time.time()) + 86_400)
    cache.write_text(json.dumps({
        "access_token": other, "refresh_token": None,
        "expiry": int(time.time()) + 86_400, "fingerprint": "someone-else",
    }))
    mine = _jwt(int(time.time()) + 86_400)
    monkeypatch.setattr(acled_auth, "_password_grant", lambda u, p: {"access_token": mine})
    assert acled_auth.get_access_token() == mine


def test_the_password_grant_is_used_directly_and_the_dead_refresh_token_is_never_tried(cache, monkeypatch):
    monkeypatch.setenv("ACLED_REFRESH_TOKEN", "expired-weeks-ago")

    def never(refresh_token):
        raise AssertionError("the refresh grant must not run when password credentials exist")

    monkeypatch.setattr(acled_auth, "_refresh_grant", never)
    token = _jwt(int(time.time()) + 86_400)
    monkeypatch.setattr(acled_auth, "_password_grant", lambda u, p: {"access_token": token})
    assert acled_auth.get_access_token() == token


def test_the_refresh_grant_remains_available_without_password_credentials(cache, monkeypatch):
    monkeypatch.delenv("ACLED_USERNAME")
    monkeypatch.delenv("ACLED_PASSWORD")
    monkeypatch.setenv("ACLED_REFRESH_TOKEN", "r0")
    token = _jwt(int(time.time()) + 86_400)
    monkeypatch.setattr(acled_auth, "_refresh_grant", lambda r: {"access_token": token, "refresh_token": "r1"})
    assert acled_auth.get_access_token() == token


def test_an_empty_cache_path_disables_the_file_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ACLED_TOKEN_CACHE_PATH", "")
    assert acled_auth._token_cache_path() is None
    acled_auth._save_file_cache(_jwt(int(time.time()) + 86_400), None)   # no error, no file


def test_the_default_cache_lives_outside_the_repository(monkeypatch):
    monkeypatch.delenv("ACLED_TOKEN_CACHE_PATH", raising=False)
    path = acled_auth._token_cache_path()
    assert path is not None
    assert "resolver" not in path.parts and "diagnostics" not in path.parts
