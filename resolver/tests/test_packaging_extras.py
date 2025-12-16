# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sanity checks for optional dependency groups used in CI."""

from importlib import metadata, util

import pytest


def test_duckdb_is_available_for_ci():
    """Fast CI should install the db extras so duckdb can be imported."""
    assert (
        util.find_spec("duckdb") is not None
    ), "duckdb module is unavailable; check the db/test extras installation"


def test_pytest_is_available_for_ci():
    """Fast CI installs the test extras, which include pytest itself."""
    assert (
        util.find_spec("pytest") is not None
    ), "pytest module missing; ensure the test extra remains correctly named"


def test_dtmapi_version_if_installed():
    """When connectors extras are installed, dtmapi should come from the 0.1.x line."""
    try:
        version = metadata.version("dtmapi")
    except metadata.PackageNotFoundError:
        pytest.skip("dtmapi not installed via connectors extra")
    major_minor = version.split(".")[:2]
    assert major_minor[0] == "0" and major_minor[1] == "1"
