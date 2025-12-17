# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import os

from fastapi import Header, HTTPException, Request


def _extract_header(request: Request, header_name: str) -> str | None:
    # Starlette's Headers object is case-insensitive; get() handles variations.
    return request.headers.get(header_name) or request.headers.get(header_name.lower())


def _env_token() -> str | None:
    return os.getenv("PYTHIA_API_TOKEN") or os.getenv("PYTHIA_API_KEY")


def _extract_bearer(value: str | None) -> str | None:
    if not value:
        return None
    if not value.lower().startswith("bearer "):
        return None
    token = value.split(" ", 1)[1].strip()
    return token or None


def require_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_pythia_token: str | None = Header(default=None, convert_underscores=False),
):
    expected = _env_token()

    # If no token is configured, allow requests (useful for local dev/testing).
    if not expected:
        return

    provided = _extract_bearer(authorization) or _extract_bearer(_extract_header(request, "Authorization"))
    if not provided:
        provided = x_pythia_token or _extract_header(request, "X-Pythia-Token")

    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
