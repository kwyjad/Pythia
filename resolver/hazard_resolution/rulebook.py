# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Load and validate ``rulebook.yaml`` — the resolution machine's config.

The rulebook is the single human-readable source of truth for resolution
behaviour. Code must read every threshold and policy switch through
:func:`load_rulebook`; hard-coding a rulebook value in Python is a bug.

Validation is strict and fails loudly: a rulebook that is missing a key,
carries a wrong type, or contains anything that looks like a credential is
rejected with a :class:`RulebookError` listing every problem at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

LOG = logging.getLogger(__name__)

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RULEBOOK_PATH = MODULE_ROOT / "rulebook.yaml"

#: Ladder rungs the machine knows how to consult. The rulebook's ``ladder``
#: list must be a non-empty, duplicate-free subset of these; its order IS the
#: precedence. GDACS is deliberately not a rung (detection + ceiling only).
KNOWN_LADDER_RUNGS = ("emdat", "reliefweb_extracted", "ifrc_go", "idmc_idu")

#: GDACS alert colours in ascending severity. Index order is meaningful.
GDACS_ALERT_LEVELS = ("green", "orange", "red")

#: Rulebook hazard names -> repo hazard codes (resolver/data/shocks.csv).
HAZARD_CODE_BY_RULEBOOK_NAME = {"cyclone": "TC", "flood": "FL", "drought": "DR"}

_KNOWN_DROUGHT_RULES = ("ipc_phase3plus_delta",)
_KNOWN_CEILING_SOURCES = ("gdacs_exposed",)
_KNOWN_CONFLICT_RULES = ("ladder_with_flag",)

#: Key-name fragments that indicate a credential. Keys come from environment
#: variables only — a rulebook carrying one is always a mistake.
_CREDENTIAL_KEY_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password")

_MISSING = object()


class RulebookError(ValueError):
    """Raised when rulebook.yaml is missing, malformed, or invalid."""


class Rulebook:
    """Read-only view over the validated rulebook mapping.

    Values are addressed with dotted paths mirroring the YAML nesting::

        rb.get("cyclone.buffer_km")   # 200
        rb.get("ladder")              # ["emdat", ...]
    """

    def __init__(self, data: Mapping[str, Any], path: Path) -> None:
        self._data = dict(data)
        self.path = path

    @property
    def raw(self) -> dict[str, Any]:
        """The underlying mapping (a shallow copy; mutating it is pointless)."""
        return dict(self._data)

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(f"rulebook has no key {dotted_key!r}")
                return default
        return node

    def __getitem__(self, dotted_key: str) -> Any:
        return self.get(dotted_key)

    def __contains__(self, dotted_key: str) -> bool:
        return self.get(dotted_key, None) is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Rulebook(path={str(self.path)!r}, keys={sorted(self._data)})"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _walk_keys(node: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            dotted = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            keys.append(dotted)
            keys.extend(_walk_keys(value, dotted))
    return keys


def validate_rulebook(data: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty when valid)."""

    problems: list[str] = []

    def _get(dotted: str) -> Any:
        node: Any = data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return _MISSING
        return node

    def _require(dotted: str) -> Any:
        value = _get(dotted)
        if value is _MISSING:
            problems.append(f"missing required key: {dotted}")
        return value

    def _require_positive_number(dotted: str) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if not _is_number(value) or value <= 0:
            problems.append(f"{dotted} must be a number > 0, got {value!r}")

    def _require_non_negative_number(dotted: str) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if not _is_number(value) or value < 0:
            problems.append(f"{dotted} must be a number >= 0, got {value!r}")

    def _require_int_in(dotted: str, lo: int, hi: int) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
            problems.append(f"{dotted} must be an integer in [{lo}, {hi}], got {value!r}")

    def _require_choice(dotted: str, choices: tuple[str, ...]) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if value not in choices:
            problems.append(f"{dotted} must be one of {list(choices)}, got {value!r}")

    def _require_bool(dotted: str) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if not isinstance(value, bool):
            problems.append(f"{dotted} must be a boolean, got {value!r}")

    if not isinstance(data, Mapping):
        return [f"rulebook must be a mapping, got {type(data).__name__}"]

    # Detection thresholds
    _require_positive_number("cyclone.buffer_km")
    _require_positive_number("cyclone.min_wind_kt")
    _require_choice("flood.gdacs_trigger_level", GDACS_ALERT_LEVELS)

    # Drought rule (thresholds are Phase 4 placeholders but must be well-typed)
    _require_choice("drought.rule", _KNOWN_DROUGHT_RULES)
    _require_int_in("drought.ipc_phase_threshold", 1, 5)
    _require_non_negative_number("drought.min_delta_people")
    _require_non_negative_number("drought.min_delta_pct")

    # Freeze policy
    freeze_days = _require("freeze_days")
    if freeze_days is not _MISSING and (
        not isinstance(freeze_days, int) or isinstance(freeze_days, bool) or freeze_days <= 0
    ):
        problems.append(f"freeze_days must be a positive integer, got {freeze_days!r}")

    # Ladder
    ladder = _require("ladder")
    if ladder is not _MISSING:
        if not isinstance(ladder, list) or not ladder:
            problems.append(f"ladder must be a non-empty list, got {ladder!r}")
        else:
            unknown = [rung for rung in ladder if rung not in KNOWN_LADDER_RUNGS]
            if unknown:
                problems.append(
                    f"ladder contains unknown rungs {unknown}; known rungs are {list(KNOWN_LADDER_RUNGS)}"
                )
            if len(set(ladder)) != len(ladder):
                problems.append(f"ladder contains duplicate rungs: {ladder}")

    # Sanity checks
    _require_choice("sanity.ceiling_source", _KNOWN_CEILING_SOURCES)
    _require_positive_number("sanity.ceiling_multiplier")
    _require_bool("sanity.population_cap")

    # Conflict handling
    _require_choice("conflict_rule", _KNOWN_CONFLICT_RULES)

    # Base-rate + backcast windows
    _require_int_in("severity_base_rate_window_start", 1950, 2100)
    backcast = _require("backcast")
    if backcast is not _MISSING:
        if not isinstance(backcast, Mapping):
            problems.append(f"backcast must be a mapping, got {backcast!r}")
        else:
            expected = set(HAZARD_CODE_BY_RULEBOOK_NAME)
            if set(backcast) != expected:
                problems.append(
                    f"backcast keys must be exactly {sorted(expected)}, got {sorted(backcast)}"
                )
            for hazard, year in backcast.items():
                if not isinstance(year, int) or isinstance(year, bool) or not 1950 <= year <= 2100:
                    problems.append(
                        f"backcast.{hazard} must be an integer year in [1950, 2100], got {year!r}"
                    )

    # Credential guard: keys come from env vars only, never from the rulebook.
    for dotted in _walk_keys(data):
        leaf = dotted.rsplit(".", 1)[-1].lower()
        if any(fragment in leaf for fragment in _CREDENTIAL_KEY_FRAGMENTS):
            problems.append(
                f"key {dotted!r} looks like a credential; API keys must come from "
                "environment variables, never the rulebook"
            )

    return problems


def load_rulebook(path: str | Path | None = None) -> Rulebook:
    """Load and validate the rulebook; raise :class:`RulebookError` on any problem."""

    rulebook_path = Path(path) if path is not None else DEFAULT_RULEBOOK_PATH
    if not rulebook_path.exists():
        raise RulebookError(f"rulebook not found at {rulebook_path}")

    try:
        with rulebook_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
    except yaml.YAMLError as exc:
        raise RulebookError(f"rulebook at {rulebook_path} is not valid YAML: {exc}") from exc

    if data is None:
        raise RulebookError(f"rulebook at {rulebook_path} is empty")

    problems = validate_rulebook(data)
    if problems:
        detail = "\n  - ".join(problems)
        raise RulebookError(
            f"rulebook at {rulebook_path} failed validation:\n  - {detail}"
        )

    LOG.debug("rulebook loaded | path=%s keys=%s", rulebook_path, sorted(data))
    return Rulebook(data, rulebook_path)
