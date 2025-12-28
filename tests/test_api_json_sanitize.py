import pytest

pytest.importorskip("duckdb")

from pythia.api import app as api_app


def test_json_sanitize_replaces_nan_and_inf():
    data = {"x": float("nan"), "y": [1, float("inf"), -float("inf")]}

    assert api_app._json_sanitize(data) == {"x": None, "y": [1, None, None]}
