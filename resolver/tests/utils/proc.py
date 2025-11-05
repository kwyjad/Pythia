"""Safe subprocess helpers for resolver's fast tests."""

from __future__ import annotations

import subprocess
from typing import Any, Mapping, Optional, Sequence

_DEFAULT_TIMEOUT = 60


def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str | bytes]:
    """Invoke ``subprocess.run`` with a safe default timeout.

    The helper mirrors :func:`subprocess.run` but ensures calls made from
    fast tests cannot hang indefinitely by applying a 60 second timeout
    unless the caller specifies an explicit ``timeout`` value.
    """

    effective_timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
    return subprocess.run(  # type: ignore[return-value]
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=effective_timeout,
        **kwargs,
    )
