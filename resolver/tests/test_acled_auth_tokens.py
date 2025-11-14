import importlib
import json
import os
from typing import Any, Dict

import pytest

import resolver.ingestion.acled_auth as acled_auth_module

_ENV_VARS = [
    "ACLED_ACCESS_TOKEN",
    "ACLED_TOKEN",
    "ACLED_REFRESH_TOKEN",
    "ACLED_USERNAME",
    "ACLED_PASSWORD",
]


def _reload(monkeypatch, **env):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.reload(acled_auth_module)
    return module
def _stub_response(status: int, payload: Dict[str, Any]) -> Any:
    class _Resp:
        status_code = status

        def __init__(self) -> None:
            self.text = json.dumps(payload)

        def json(self) -> Dict[str, Any]:
            return payload

    return _Resp()


def test_get_access_token_accepts_opaque_token(monkeypatch):
    module = _reload(monkeypatch, ACLED_ACCESS_TOKEN="opaque-token")

    def _fail(*_args, **_kwargs):  # pragma: no cover - defensive helper
        raise AssertionError("refresh/password grant should not run for opaque tokens")

    monkeypatch.setattr(module, "_refresh_grant", _fail)
    monkeypatch.setattr(module, "_password_grant", _fail)

    token = module.get_access_token()

    assert token == "opaque-token"


def test_get_access_token_uses_legacy_env_var(monkeypatch):
    module = _reload(monkeypatch, ACLED_TOKEN="legacy-token")

    token = module.get_access_token()

    assert token == "legacy-token"
    assert os.environ.get("ACLED_ACCESS_TOKEN") == "legacy-token"


def test_refresh_grant_fetches_new_tokens(monkeypatch):
    module = _reload(monkeypatch, ACLED_REFRESH_TOKEN="refresh-123")

    def _fake_refresh(refresh_token: str) -> Dict[str, str]:
        assert refresh_token == "refresh-123"
        return {"access_token": "new-token", "refresh_token": "new-refresh"}

    monkeypatch.setattr(module, "_refresh_grant", _fake_refresh)

    token = module.get_access_token()

    assert token == "new-token"
    assert os.environ.get("ACLED_ACCESS_TOKEN") == "new-token"
    assert os.environ.get("ACLED_REFRESH_TOKEN") == "new-refresh"


def test_password_grant_fetches_when_no_refresh(monkeypatch):
    captured: Dict[str, Any] = {}

    module = _reload(monkeypatch, ACLED_USERNAME="user@example.com", ACLED_PASSWORD="secret")

    def _fake_post(url, data, headers, timeout):
        captured.update({"url": url, "data": dict(data), "headers": headers, "timeout": timeout})
        return _stub_response(200, {"access_token": "token-xyz", "refresh_token": "refresh-xyz"})

    monkeypatch.setattr(module.requests, "post", _fake_post)

    token = module.get_access_token()

    assert token == "token-xyz"
    assert os.environ.get("ACLED_REFRESH_TOKEN") == "refresh-xyz"
    assert captured["url"] == module.OAUTH_TOKEN_URL
    assert captured["data"]["grant_type"] == "password"
    assert captured["data"]["client_id"] == module.OAUTH_CLIENT_ID
    assert captured["data"]["username"] == "user@example.com"
    assert captured["data"]["password"] == "secret"


def test_password_grant_raises_on_non_200(monkeypatch):
    module = _reload(monkeypatch, ACLED_USERNAME="user@example.com", ACLED_PASSWORD="secret")

    def _fake_post(*_args, **_kwargs):
        return _stub_response(401, {"error": "invalid"})

    monkeypatch.setattr(module.requests, "post", _fake_post)

    with pytest.raises(RuntimeError, match="password grant failed: status=401"):
        module._password_grant("user@example.com", "secret")


def test_refresh_grant_raises_on_non_200(monkeypatch):
    module = _reload(monkeypatch, ACLED_REFRESH_TOKEN="refresh-123")

    def _fake_post(*_args, **_kwargs):
        return _stub_response(500, {"error": "server"})

    monkeypatch.setattr(module.requests, "post", _fake_post)

    with pytest.raises(RuntimeError, match="refresh grant failed: status=500"):
        module._refresh_grant("refresh-123")
