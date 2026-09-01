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

#: IBTrACS wind columns detection may consult (rulebook order = priority).
KNOWN_WIND_SOURCES = ("usa_wind", "wmo_wind")

#: IBTrACS CSV files the connector knows how to fetch.
KNOWN_IBTRACS_SCOPES = ("last3years", "ALL")

#: File shapes the Dartmouth Flood Observatory archive reader can parse.
KNOWN_DFO_FORMATS = ("xlsx", "csv")

#: Rulebook hazard names -> repo hazard codes (resolver/data/shocks.csv).
HAZARD_CODE_BY_RULEBOOK_NAME = {"cyclone": "TC", "flood": "FL", "drought": "DR"}

#: Repo hazard codes -> rulebook hazard names (the inverse; connectors and
#: the reconciler address the rulebook by hazard CODE).
RULEBOOK_NAME_BY_HAZARD_CODE = {
    code: name for name, code in HAZARD_CODE_BY_RULEBOOK_NAME.items()
}

#: Hazards whose detection layer runs a ReliefWeb silence sweep before a
#: zero. Drought resolves from IPC instead (Phase 4), so it has no sweep.
SWEEP_HAZARD_KEYS = ("cyclone", "flood")

_KNOWN_DROUGHT_RULES = ("ipc_phase3plus_delta",)

#: How a drought delta maps onto the months its IPC analysis window covers.
_KNOWN_DROUGHT_MONTH_ATTRIBUTION = ("analysis_window", "start_month")

#: Acquisition paths for IPC analyses, in the order ``source_priority`` may
#: list them. ``ipc_api`` is the IPC public API; ``facts_resolved`` reuses the
#: Phase 3+ rows the repo's FEWS NET / IPC connectors already write.
_KNOWN_IPC_SOURCES = ("ipc_api", "facts_resolved")

#: Drought-indicator feed shapes the machine knows how to parse.
_KNOWN_INDICATOR_PROVIDERS = ("asap", "tabular")
_KNOWN_INDICATOR_COMBINE = ("any", "all")
_KNOWN_INDICATOR_DIRECTIONS = ("below", "above")
_KNOWN_CEILING_SOURCES = ("gdacs_exposed",)
_KNOWN_CONFLICT_RULES = ("ladder_with_flag",)
_KNOWN_DETECTION_ATTRIBUTION = ("overlap",)
_KNOWN_FIGURE_ATTRIBUTION = ("start_month",)

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

    def _require_str_list(dotted: str, choices: tuple[str, ...] | None = None) -> None:
        value = _require(dotted)
        if value is _MISSING:
            return
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            problems.append(f"{dotted} must be a non-empty list of strings, got {value!r}")
            return
        if choices is not None:
            unknown = [item for item in value if item not in choices]
            if unknown:
                problems.append(
                    f"{dotted} contains unknown entries {unknown}; known: {list(choices)}"
                )
        if len(set(value)) != len(value):
            problems.append(f"{dotted} contains duplicates: {value}")

    def _require_hazard_keyed_str_lists(dotted: str) -> None:
        """A mapping of repo hazard code -> non-empty list of strings."""
        value = _require(dotted)
        if value is _MISSING:
            return
        if not isinstance(value, Mapping) or not value:
            problems.append(f"{dotted} must be a non-empty mapping, got {value!r}")
            return
        known = set(HAZARD_CODE_BY_RULEBOOK_NAME.values())
        unknown = sorted(set(value) - known)
        if unknown:
            problems.append(
                f"{dotted} has unknown hazard codes {unknown}; known: {sorted(known)}"
            )
        for hazard, items in value.items():
            if (
                not isinstance(items, list)
                or not items
                or not all(isinstance(i, str) and i.strip() for i in items)
            ):
                problems.append(
                    f"{dotted}.{hazard} must be a non-empty list of strings, got {items!r}"
                )

    def _require_fetch_window(prefix: str) -> None:
        """lookback/lookahead month counts framing a source's fetch window."""
        _require_int_in(f"{prefix}.lookback_months", 0, 120)
        _require_int_in(f"{prefix}.lookahead_months", 0, 120)

    # Detection thresholds
    _require_positive_number("cyclone.buffer_km")
    _require_positive_number("cyclone.min_wind_kt")
    _require_str_list("cyclone.wind_source_priority", KNOWN_WIND_SOURCES)
    _require_choice("flood.gdacs_trigger_level", GDACS_ALERT_LEVELS)

    # GDACS usage (flood detection input; also the sanity ceiling source).
    # Endpoints belong to resolver/connectors/gdacs.py — see the rulebook
    # comment; what is configured here is window, pacing and zero safety.
    _require_non_negative_number("flood.gdacs.request_delay_sec")
    _require_int_in("flood.gdacs.enrich_workers", 1, 32)
    _require_fetch_window("flood.gdacs")
    _require_int_in("flood.gdacs.coverage_grace_days", 0, 60)

    # IBTrACS connector (cyclone detection input)
    url_template = _require("cyclone.ibtracs.url_template")
    if url_template is not _MISSING and (
        not isinstance(url_template, str) or "{scope}" not in url_template
    ):
        problems.append(
            f"cyclone.ibtracs.url_template must be a string containing '{{scope}}', got {url_template!r}"
        )
    _require_choice("cyclone.ibtracs.default_scope", KNOWN_IBTRACS_SCOPES)
    _require_positive_number("cyclone.ibtracs.request_timeout_sec")
    _require_int_in("cyclone.ibtracs.coverage_grace_days", 0, 60)

    # ReliefWeb silence sweep (zero-resolution gate) — every sweeping hazard
    # carries the same block, so a new hazard cannot be half-configured.
    for hazard_key in SWEEP_HAZARD_KEYS:
        sweep = f"{hazard_key}.reliefweb_sweep"
        _require_str_list(f"{sweep}.disaster_types")
        _require_str_list(f"{sweep}.keywords")
        _require_int_in(f"{sweep}.publication_pad_days", 0, 90)
        _require_int_in(f"{sweep}.max_hits_for_silence", 0, 100)
        _require_int_in(f"{sweep}.sample_size", 1, 50)
        _require_positive_number(f"{sweep}.request_timeout_sec")
        _require_non_negative_number(f"{sweep}.request_delay_sec")
    api_base_url = _require("reliefweb.api_base_url")
    if api_base_url is not _MISSING and (
        not isinstance(api_base_url, str) or not api_base_url.startswith("https://")
    ):
        problems.append(
            f"reliefweb.api_base_url must be an https:// URL, got {api_base_url!r}"
        )

    # ReliefWeb document selection (ladder rung 2 input).
    _require_str_list("reliefweb.documents.formats")
    _require_int_in("reliefweb.documents.max_docs_per_cell", 1, 200)
    _require_int_in("reliefweb.documents.candidate_pool_size", 1, 1000)
    _require_int_in("reliefweb.documents.publication_pad_days", 0, 90)
    _require_str_list("reliefweb.documents.source_priority")
    _require_int_in("reliefweb.documents.body_char_limit", 500, 500_000)
    _require_positive_number("reliefweb.documents.request_timeout_sec")
    _require_non_negative_number("reliefweb.documents.request_delay_sec")
    # A pool no larger than the cap makes the source ranking a no-op: every
    # document fetched is read, so "most authoritative first" degenerates
    # into "most recent first" without anything failing.
    pool = _get("reliefweb.documents.candidate_pool_size")
    cap = _get("reliefweb.documents.max_docs_per_cell")
    if isinstance(pool, int) and isinstance(cap, int) and pool < cap:
        problems.append(
            "reliefweb.documents.candidate_pool_size must be >= "
            f"max_docs_per_cell, got {pool} < {cap} (ranking would never "
            "choose between documents)"
        )

    # Figure attribution precedence: an ordered list of named tiers.
    precedence = _require("reliefweb.authority_precedence")
    if precedence is not _MISSING:
        if not isinstance(precedence, list) or not precedence:
            problems.append(
                f"reliefweb.authority_precedence must be a non-empty list, got {precedence!r}"
            )
        else:
            seen_tiers: list[str] = []
            for index, entry in enumerate(precedence):
                where = f"reliefweb.authority_precedence[{index}]"
                if not isinstance(entry, Mapping):
                    problems.append(f"{where} must be a mapping, got {entry!r}")
                    continue
                tier = entry.get("tier")
                if not isinstance(tier, str) or not tier.strip():
                    problems.append(f"{where}.tier must be a non-empty string, got {tier!r}")
                else:
                    seen_tiers.append(tier)
                keywords = entry.get("keywords")
                if (
                    not isinstance(keywords, list)
                    or not keywords
                    or not all(isinstance(k, str) and k.strip() for k in keywords)
                ):
                    problems.append(
                        f"{where}.keywords must be a non-empty list of strings, got {keywords!r}"
                    )
            if len(set(seen_tiers)) != len(seen_tiers):
                problems.append(
                    f"reliefweb.authority_precedence has duplicate tiers: {seen_tiers}"
                )

    # Households -> people conversion.
    _require_positive_number("reliefweb.household_conversion.default_multiplier")
    by_iso3 = _require("reliefweb.household_conversion.by_iso3")
    if by_iso3 is not _MISSING:
        if not isinstance(by_iso3, Mapping):
            problems.append(
                f"reliefweb.household_conversion.by_iso3 must be a mapping, got {by_iso3!r}"
            )
        else:
            for iso3, multiplier in by_iso3.items():
                if not isinstance(iso3, str) or len(iso3) != 3 or not iso3.isalpha():
                    problems.append(
                        f"reliefweb.household_conversion.by_iso3 key {iso3!r} is not an ISO3 code"
                    )
                if not _is_number(multiplier) or multiplier <= 0:
                    problems.append(
                        f"reliefweb.household_conversion.by_iso3.{iso3} must be a "
                        f"number > 0, got {multiplier!r}"
                    )

    # LLM extraction policy. The MODEL is not configured here: the rulebook
    # names a role and pythia/config.yaml resolves it, so model choice keeps
    # one source of truth.
    _require_bool("extraction.enabled")
    model_role = _require("extraction.model_role")
    if model_role is not _MISSING and (
        not isinstance(model_role, str) or not model_role.strip()
    ):
        problems.append(
            f"extraction.model_role must be a non-empty role name, got {model_role!r}"
        )
    elif isinstance(model_role, str) and ":" in model_role:
        problems.append(
            f"extraction.model_role must be a ROLE name, not a model ref ({model_role!r}); "
            "model ids live in pythia/config.yaml's llm.models registry"
        )
    prompt_version = _require("extraction.prompt_version")
    if prompt_version is not _MISSING and (
        not isinstance(prompt_version, str) or not prompt_version.strip()
    ):
        problems.append(
            f"extraction.prompt_version must be a non-empty string, got {prompt_version!r}"
        )
    _require_non_negative_number("extraction.temperature")
    _require_int_in("extraction.max_output_tokens", 64, 32_000)
    _require_positive_number("extraction.request_timeout_sec")
    _require_int_in("extraction.max_calls_per_month", 0, 1_000_000)
    _require_int_in("extraction.backcast_max_calls_per_month", 0, 1_000_000)
    _sub_cap = _get("extraction.backcast_max_calls_per_month")
    _total_cap = _get("extraction.max_calls_per_month")
    if (
        isinstance(_sub_cap, int)
        and isinstance(_total_cap, int)
        and _sub_cap > _total_cap
    ):
        problems.append(
            "extraction.backcast_max_calls_per_month must not exceed "
            "extraction.max_calls_per_month — the sub-cap carves a backcast "
            "share OUT of the monthly total, it does not add to it"
        )
    _require_bool("extraction.skip_when_higher_rung_populated")

    # Drought rule: IPC Phase 3+ delta, gated by drought indicators.
    _require_choice("drought.rule", _KNOWN_DROUGHT_RULES)
    _require_int_in("drought.ipc_phase_threshold", 1, 5)
    _require_non_negative_number("drought.min_delta_people")
    _require_non_negative_number("drought.min_delta_pct")
    _require_choice("drought.month_attribution", _KNOWN_DROUGHT_MONTH_ATTRIBUTION)

    ipc_url = _require("drought.ipc.api_url")
    if ipc_url is not _MISSING and (
        not isinstance(ipc_url, str) or not ipc_url.startswith("https://")
    ):
        problems.append(f"drought.ipc.api_url must be an https:// URL, got {ipc_url!r}")
    _require_positive_number("drought.ipc.request_timeout_sec")
    _require_fetch_window("drought.ipc")
    _require_bool("drought.ipc.use_facts_resolved")
    metric = _require("drought.ipc.facts_resolved_metric")
    if metric is not _MISSING and (not isinstance(metric, str) or not metric.strip()):
        problems.append(
            f"drought.ipc.facts_resolved_metric must be a non-empty string, got {metric!r}"
        )
    _require_str_list("drought.ipc.facts_resolved_publishers")
    _require_str_list("drought.ipc.source_priority", _KNOWN_IPC_SOURCES)
    # The rule is a DIFFERENCE between two analyses, so the fetch window has
    # to be able to contain both. IPC publishes one to three analyses a year;
    # a window shorter than a year routinely holds only the covering analysis,
    # which yields no baseline and therefore no answer — silently.
    lookback = _get("drought.ipc.lookback_months")
    if isinstance(lookback, int) and not isinstance(lookback, bool) and lookback < 12:
        problems.append(
            f"drought.ipc.lookback_months must be >= 12, got {lookback} — the rule "
            "diffs against the PREVIOUS analysis, and IPC publishes too sparsely "
            "for a shorter window to reliably contain one"
        )

    # Drought indicators: the attribution gate and the zero gate.
    _require_choice("drought.indicators.combine", _KNOWN_INDICATOR_COMBINE)
    _require_int_in("drought.indicators.max_observation_age_months", 0, 60)
    entries = _require("drought.indicators.entries")
    if entries is not _MISSING:
        if not isinstance(entries, list) or not entries:
            problems.append(
                f"drought.indicators.entries must be a non-empty list, got {entries!r}"
            )
        else:
            names: list[str] = []
            any_required = False
            for index, entry in enumerate(entries):
                where = f"drought.indicators.entries[{index}]"
                if not isinstance(entry, Mapping):
                    problems.append(f"{where} must be a mapping, got {entry!r}")
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name.strip():
                    problems.append(f"{where}.name must be a non-empty string, got {name!r}")
                else:
                    names.append(name)
                provider = entry.get("provider")
                if provider not in _KNOWN_INDICATOR_PROVIDERS:
                    problems.append(
                        f"{where}.provider must be one of "
                        f"{list(_KNOWN_INDICATOR_PROVIDERS)}, got {provider!r}"
                    )
                url = entry.get("url")
                if not isinstance(url, str):
                    problems.append(
                        f"{where}.url must be a string (empty means 'not consulted'), got {url!r}"
                    )
                elif url.strip() and not url.startswith("https://"):
                    problems.append(f"{where}.url must be an https:// URL, got {url!r}")
                required = entry.get("required")
                if not isinstance(required, bool):
                    problems.append(f"{where}.required must be a boolean, got {required!r}")
                else:
                    any_required = any_required or required
                timeout = entry.get("request_timeout_sec")
                if not _is_number(timeout) or timeout <= 0:
                    problems.append(
                        f"{where}.request_timeout_sec must be a number > 0, got {timeout!r}"
                    )
                # A required entry with no url can never be read, so it would
                # suppress every zero for as long as it stayed misconfigured.
                if required is True and isinstance(url, str) and not url.strip():
                    problems.append(
                        f"{where} is required: true but has an empty url — it could "
                        "never be read, and an unreadable required indicator "
                        "suppresses every drought zero"
                    )
                if provider == "asap":
                    classes = entry.get("drought_classes")
                    if (
                        not isinstance(classes, list)
                        or not classes
                        or not all(str(c).strip() for c in classes)
                    ):
                        problems.append(
                            f"{where}.drought_classes must be a non-empty list, got {classes!r}"
                        )
                if provider == "tabular":
                    if not _is_number(entry.get("threshold")):
                        problems.append(
                            f"{where}.threshold must be a number, got {entry.get('threshold')!r}"
                        )
                    if entry.get("direction") not in _KNOWN_INDICATOR_DIRECTIONS:
                        problems.append(
                            f"{where}.direction must be one of "
                            f"{list(_KNOWN_INDICATOR_DIRECTIONS)}, got {entry.get('direction')!r}"
                        )
            if len(set(names)) != len(names):
                problems.append(f"drought.indicators.entries has duplicate names: {names}")
            # With nothing required, an outage in every feed reads as "no
            # drought anywhere" and the machine would zero the whole world.
            if not any_required:
                problems.append(
                    "drought.indicators.entries must mark at least one entry "
                    "required: true — with none, a total indicator outage would "
                    "be indistinguishable from a quiet month and zero every country"
                )

    # Freeze policy
    freeze_days = _require("freeze_days")
    if freeze_days is not _MISSING and (
        not isinstance(freeze_days, int) or isinstance(freeze_days, bool) or freeze_days <= 0
    ):
        problems.append(f"freeze_days must be a positive integer, got {freeze_days!r}")

    # Event-to-month attribution
    _require_choice("event_attribution.detection", _KNOWN_DETECTION_ATTRIBUTION)
    _require_choice("event_attribution.figure", _KNOWN_FIGURE_ATTRIBUTION)

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

    # Lower-bound rungs must be rungs the ladder actually walks, otherwise a
    # displacement figure could be published as a people-affected figure.
    lower_bound = _require("lower_bound_rungs")
    if lower_bound is not _MISSING:
        if not isinstance(lower_bound, list):
            problems.append(f"lower_bound_rungs must be a list, got {lower_bound!r}")
        else:
            unknown = [r for r in lower_bound if r not in KNOWN_LADDER_RUNGS]
            if unknown:
                problems.append(
                    f"lower_bound_rungs contains unknown rungs {unknown}; "
                    f"known rungs are {list(KNOWN_LADDER_RUNGS)}"
                )
            if len(set(lower_bound)) != len(lower_bound):
                problems.append(f"lower_bound_rungs contains duplicates: {lower_bound}")
            if isinstance(ladder, list):
                orphans = [r for r in lower_bound if r not in ladder]
                if orphans:
                    problems.append(
                        f"lower_bound_rungs entries {orphans} are not in the ladder; "
                        "a lower-bound rung the ladder never walks is a no-op"
                    )

    # Sanity checks
    _require_choice("sanity.ceiling_source", _KNOWN_CEILING_SOURCES)
    _require_positive_number("sanity.ceiling_multiplier")
    _require_bool("sanity.population_cap")

    # Conflict handling
    _require_choice("conflict_rule", _KNOWN_CONFLICT_RULES)
    oom = _require("conflict_detection.order_of_magnitude_factor")
    if oom is not _MISSING and (not _is_number(oom) or oom <= 1):
        problems.append(
            "conflict_detection.order_of_magnitude_factor must be a number > 1, "
            f"got {oom!r} (a factor <= 1 would flag every pair of rungs)"
        )

    # Ladder source connectors. Endpoints live here; credentials never do
    # (the credential guard below rejects any key that looks like one).
    emdat_url = _require("emdat.api_url")
    if emdat_url is not _MISSING and (
        not isinstance(emdat_url, str) or not emdat_url.startswith("https://")
    ):
        problems.append(f"emdat.api_url must be an https:// URL, got {emdat_url!r}")
    _require_positive_number("emdat.request_timeout_sec")
    _require_int_in("emdat.page_limit", 1, 10000)
    _require_int_in("emdat.max_pages", 1, 1000)
    _require_hazard_keyed_str_lists("emdat.classif_keys")
    _require_fetch_window("emdat")

    go_url = _require("ifrc_go.api_base_url")
    if go_url is not _MISSING and (
        not isinstance(go_url, str) or not go_url.startswith("https://")
    ):
        problems.append(f"ifrc_go.api_base_url must be an https:// URL, got {go_url!r}")
    go_path = _require("ifrc_go.field_report_path")
    if go_path is not _MISSING and (not isinstance(go_path, str) or not go_path.strip()):
        problems.append(f"ifrc_go.field_report_path must be a non-empty string, got {go_path!r}")
    _require_int_in("ifrc_go.page_size", 1, 1000)
    _require_int_in("ifrc_go.max_pages", 1, 1000)
    _require_positive_number("ifrc_go.request_timeout_sec")
    _require_str_list("ifrc_go.affected_fields")
    _require_fetch_window("ifrc_go")

    idu_url = _require("idmc_idu.api_url")
    if idu_url is not _MISSING and (
        not isinstance(idu_url, str) or not idu_url.startswith("https://")
    ):
        problems.append(f"idmc_idu.api_url must be an https:// URL, got {idu_url!r}")
    _require_positive_number("idmc_idu.request_timeout_sec")
    _require_str_list("idmc_idu.displacement_types")
    _require_hazard_keyed_str_lists("idmc_idu.hazard_keywords")
    _require_fetch_window("idmc_idu")

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

    # Raw-cache retention. keep_revisions_per_record is capped low on
    # purpose: these caches hold document bodies, and "keep a few revisions"
    # is how an append-only log gets rebuilt one increment at a time.
    _require_int_in("raw_cache.keep_revisions_per_record", 1, 10)
    ratio = _require("raw_cache.compaction.file_bloat_ratio")
    if ratio is not _MISSING and (not _is_number(ratio) or ratio < 1.0):
        problems.append(
            "raw_cache.compaction.file_bloat_ratio must be a number >= 1.0, "
            f"got {ratio!r}"
        )

    # Base-rate publication thresholds.
    _require_int_in("base_rates.occurrence.min_years", 1, 200)
    _require_int_in("base_rates.severity.min_events", 1, 10_000)

    # Prompt-block rendering (Phase 6). The block is generated from this
    # file, so these are the only two knobs that change what a forecaster
    # sees without a code change.
    _require_int_in("base_rates.prompt.min_events_for_quantiles", 1, 10_000)
    _require_int_in("base_rates.prompt.max_chars", 200, 20_000)

    # Dartmouth Flood Observatory cross-check (calibration only).
    _require_bool("dfo.enabled")
    dfo_url = _require("dfo.url")
    if dfo_url is not _MISSING and (
        not isinstance(dfo_url, str) or not dfo_url.startswith("https://")
    ):
        problems.append(f"dfo.url must be an https:// URL, got {dfo_url!r}")
    _require_choice("dfo.format", KNOWN_DFO_FORMATS)
    _require_positive_number("dfo.request_timeout_sec")
    _require_int_in("dfo.cross_check_end_year", 1950, 2100)
    _require_int_in("dfo.min_years_for_comparison", 1, 200)
    dfo_divergence = _require("dfo.divergence_factor")
    if dfo_divergence is not _MISSING and (
        not _is_number(dfo_divergence) or dfo_divergence <= 1
    ):
        problems.append(
            "dfo.divergence_factor must be a number > 1, got "
            f"{dfo_divergence!r} (a factor <= 1 would flag every country)"
        )
    # The cross-check earns its name by looking at years the machine did
    # NOT backcast. Overlapping the windows would compare the machine's
    # flood detection against an archive of the same floods it already
    # resolved from, which confirms nothing.
    dfo_end = _get("dfo.cross_check_end_year")
    flood_start = _get("backcast.flood")
    if (
        isinstance(dfo_end, int)
        and not isinstance(dfo_end, bool)
        and isinstance(flood_start, int)
        and not isinstance(flood_start, bool)
        and dfo_end > flood_start
    ):
        problems.append(
            f"dfo.cross_check_end_year ({dfo_end}) must be <= backcast.flood "
            f"({flood_start}) — the cross-check compares the machine's era "
            "against the years BEFORE it, and overlapping windows would "
            "compare the machine against itself"
        )

    # Credential guard: keys come from env vars only, never from the rulebook.
    #
    # The VALUE has to be a string for this to fire. A credential is always
    # text; a numeric setting whose name happens to contain a credential
    # fragment ("extraction.max_output_tokens") is a token BUDGET, not a
    # token. Without the type test the guard rejects perfectly good config,
    # and the fix people reach for is renaming the setting — which weakens
    # the rulebook's readability to satisfy a check that was wrong.
    for dotted in _walk_keys(data):
        leaf = dotted.rsplit(".", 1)[-1].lower()
        if not any(fragment in leaf for fragment in _CREDENTIAL_KEY_FRAGMENTS):
            continue
        value = _get(dotted)
        if isinstance(value, str):
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
