# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guard the Codex canary gate test roster."""

from resolver.ci.canary import CANARY_TESTS


def test_canary_tests_are_stable() -> None:
    expected = [
        "resolver/tests/test_run_connectors_extra_args.py::test_run_connectors_passes_extra_args_and_env",
        "resolver/tests/test_iso_normalize_and_drop_reasons.py::test_dtm_drop_reason_counters_capture_iso_and_value_failures",
        "resolver/tests/test_dtm_soft_timeouts.py::test_soft_timeouts_yield_ok_empty",
    ]
    assert CANARY_TESTS == expected
