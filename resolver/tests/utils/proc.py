# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Safe subprocess helpers for resolver's fast tests."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Mapping, Optional, Sequence

_ENV_DEFAULTS: Mapping[str, str] = {
    "IDMC_NETWORK_MODE": "fixture",
    "IDMC_ALLOW_HDX_FALLBACK": "0",
    "PYTEST_PROC_TIMEOUT": "60",
}


def _resolve_timeout(explicit_timeout: Optional[float]) -> float:
    """Return the timeout to apply to subprocess calls."""

    if explicit_timeout is not None:
        return float(explicit_timeout)
    env_timeout = os.environ.get("PYTEST_PROC_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            pass
    return float(_ENV_DEFAULTS["PYTEST_PROC_TIMEOUT"])


def _prepare_env(overrides: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    """Merge resolver fast-test defaults with caller overrides."""

    merged = dict(os.environ)
    for key, value in _ENV_DEFAULTS.items():
        merged.setdefault(key, value)
    if overrides:
        merged.update(overrides)
    return merged


def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
    timeout: Optional[int | float] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str | bytes]:
    """Invoke ``subprocess.run`` with a safe default timeout.

    The helper mirrors :func:`subprocess.run` but ensures calls made from
    fast tests cannot hang indefinitely by applying a sensible timeout
    unless the caller specifies an explicit ``timeout`` value.  Environment
    defaults force offline-friendly behaviours unless a test overrides
    them explicitly.
    """

    effective_timeout = _resolve_timeout(timeout)
    prepared_env = _prepare_env(env)
    return subprocess.run(  # type: ignore[return-value]
        cmd,
        cwd=cwd,
        env=prepared_env,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=effective_timeout,
        **kwargs,
    )
