# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ensure the resolver package can be imported after editable installs."""

def test_import_resolver():
    import resolver  # noqa: F401
