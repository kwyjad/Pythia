import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.scenario_writer as scenario_writer


def test_safe_json_loads_scenario_handles_code_fence() -> None:
    fenced = """```json
    {"primary": {"bucket_label": "bucket_3", "probability": 0.6, "text": "Test"},
     "alternative": null}
    ```"""

    obj = scenario_writer._safe_json_loads_scenario(fenced)

    assert obj["primary"]["bucket_label"] == "bucket_3"
    assert obj["primary"]["probability"] == 0.6
