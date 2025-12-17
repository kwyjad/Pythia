# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import Any, Dict, Frozenset, Iterable, Tuple

import requests
import streamlit as st


def get_base_url() -> str:
    """Resolve the API base URL from secrets or environment."""

    secret_base = None
    try:
        secret_base = st.secrets.get("PYTHIA_API_BASE") or st.secrets.get("pythia_api_base")
    except Exception:
        secret_base = None

    env_base = os.getenv("PYTHIA_API_BASE") or os.getenv("pythia_api_base")
    return (secret_base or env_base or "http://localhost:8080/v1").rstrip("/")


def get_token() -> str:
    """Resolve the bearer token from secrets or environment."""

    secret_token = None
    try:
        secret_token = st.secrets.get("PYTHIA_API_TOKEN") or st.secrets.get("pythia_api_token")
    except Exception:
        secret_token = None

    env_token = os.getenv("PYTHIA_API_TOKEN") or os.getenv("pythia_api_token")
    return secret_token or env_token or ""


def _build_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")

    if path.startswith("http"):
        return path

    normalized_path = path if path.startswith("/") else f"/{path}"
    if base.endswith("/v1") and normalized_path.startswith("/v1"):
        normalized_path = normalized_path[len("/v1") :]

    return f"{base}{normalized_path}"


def _build_headers(token: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_url(path: str) -> str:
    return _build_url(get_base_url(), path)


def _handle_response(response: requests.Response, path: str) -> Dict[str, Any]:
    if not response.ok:
        detail = response.text or response.reason
        message = f"API request to {path} failed ({response.status_code}): {detail}"
        st.error(message)
        raise RuntimeError(message)

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard for UI
        message = f"API request to {path} returned non-JSON content."
        st.error(message)
        raise RuntimeError(message) from exc


@st.cache_data(show_spinner=False)
def _cached_get(base_url: str, path: str, param_items: Frozenset[Tuple[str, Any]], token: str) -> Dict[str, Any]:
    params = {k: v for k, v in sorted(param_items) if v not in (None, "")}
    url = _build_url(base_url, path)
    response = requests.get(url, params=params, headers=_build_headers(token), timeout=30)
    return _handle_response(response, path)


def api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Cached GET wrapper against the Pythia API."""

    base_url = get_base_url()
    token = get_token()
    param_items: Frozenset[Tuple[str, Any]] = frozenset((params or {}).items())
    return _cached_get(base_url, path, param_items, token)


def api_post(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """POST wrapper against the Pythia API."""

    base_url = get_base_url()
    token = get_token()
    url = _build_url(base_url, path)
    response = requests.post(url, json=payload or {}, headers=_build_headers(token), timeout=30)
    return _handle_response(response, path)


def clear_cache() -> None:
    _cached_get.clear()
