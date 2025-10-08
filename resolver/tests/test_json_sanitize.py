import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from resolver.utils.json_sanitize import json_default


def test_json_default_serialises_common_pandas_numpy_types():
    payload = {
        "timestamp": pd.Timestamp("2024-02-28"),
        "datetime": dt.datetime(2024, 2, 28, 12, 30, 45),
        "date": dt.date(2024, 2, 27),
        "integer": np.int64(5),
        "floating": np.float64(3.14),
        "boolean": np.bool_(True),
        "timedelta": pd.Timedelta(days=2),
        "array": np.array([1, 2, 3]),
    }

    encoded = json.dumps(payload, default=json_default, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["timestamp"] == "2024-02-28"
    assert decoded["datetime"] == "2024-02-28T12:30:45"
    assert decoded["date"] == "2024-02-27"
    assert decoded["integer"] == 5
    assert decoded["floating"] == pytest.approx(3.14)
    assert decoded["boolean"] is True
    assert decoded["timedelta"] == "P2DT0H0M0S"
    assert decoded["array"] == [1, 2, 3]


def test_json_default_handles_nat_and_timezone():
    tz_timestamp = pd.Timestamp("2024-02-28T00:00:00Z")
    assert json_default(tz_timestamp) == "2024-02-28"
    assert json_default(pd.NaT) is None
