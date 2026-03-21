# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Single source of truth for the PYTHIA_TEST_MODE flag."""

import os


def is_test_mode() -> bool:
    """Return True when the current pipeline run is a test run."""
    return os.environ.get("PYTHIA_TEST_MODE", "").strip() in ("1", "true", "yes")
