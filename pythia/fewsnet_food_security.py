# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FEWS NET Food Security — backward-compatibility shim.

.. deprecated::
    Use :mod:`pythia.food_security` instead.  This module re-exports the
    unified loader and formatters for existing callers.
"""

from __future__ import annotations

from typing import Any

from pythia.food_security import (
    format_food_security_for_prompt,
    format_food_security_for_spd,
    load_food_security,
)


def load_fewsnet_food_security(
    iso3: str,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Load food security data (FEWS NET or IPC fallback).

    .. deprecated:: Use :func:`pythia.food_security.load_food_security`.
    """
    return load_food_security(iso3, db_url=db_url)


def format_fewsnet_for_prompt(data: dict[str, Any] | None) -> str:
    """Format food security data for RC/triage prompts.

    .. deprecated:: Use :func:`pythia.food_security.format_food_security_for_prompt`.
    """
    return format_food_security_for_prompt(data)


def format_fewsnet_for_spd(data: dict[str, Any] | None) -> str:
    """Format food security data for SPD prompts.

    .. deprecated:: Use :func:`pythia.food_security.format_food_security_for_spd`.
    """
    return format_food_security_for_spd(data)
