# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from resolver.ingestion._exit_policy import compute_exit_code


def test_config_disabled_is_success() -> None:
    results = [{"status": "skipped", "reason": "disabled: config"}]
    assert compute_exit_code(results) == 0


def test_missing_secret_causes_failure() -> None:
    results = [{"status": "skipped", "reason": "missing: secret ACLED_TOKEN"}]
    assert compute_exit_code(results) == 1


def test_success_when_any_connector_runs() -> None:
    results = [
        {"status": "skipped", "reason": "disabled: config"},
        {"status": "ok", "reason": None},
    ]
    assert compute_exit_code(results) == 0


def test_error_forces_failure() -> None:
    results = [
        {"status": "ok", "reason": None},
        {"status": "error", "reason": None},
    ]
    assert compute_exit_code(results) == 1
