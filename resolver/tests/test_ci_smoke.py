# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Fast smoke test to ensure CI always has at least one test to run."""

import importlib


def test_ci_smoke() -> None:
    """Verify the resolver package can be imported and exposes a package attribute."""
    module = importlib.import_module("resolver")
    assert hasattr(module, "__package__")
