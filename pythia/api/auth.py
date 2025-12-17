# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from fastapi import Header, HTTPException, Request
from pythia.config import load as load_cfg


def _extract_header(request: Request, header_name: str) -> str | None:
    # Starlette's Headers object is case-insensitive; get() handles variations.
    return request.headers.get(header_name) or request.headers.get(header_name.lower())


def require_token(
    request: Request, x_pythia_token: str | None = Header(default=None, convert_underscores=False)
):
    cfg = load_cfg() or {}
    security_cfg = cfg.get("security") or {}
    want = set(security_cfg.get("api_tokens") or [])
    header_name = security_cfg.get("api_token_header") or "X-Pythia-Token"

    provided = x_pythia_token or _extract_header(request, header_name) or _extract_header(
        request, "X-Pythia-Token"
    )

    if not provided or provided not in want:
        raise HTTPException(status_code=401, detail="Unauthorized")
