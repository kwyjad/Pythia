import math

import pytest

from forecaster import scoring


@pytest.mark.parametrize(
    "metric,value,expected",
    [
        ("PA", 0, 1),
        ("PA", 9999, 1),
        ("PA", 10000, 2),
        ("PA", 49999, 2),
        ("PA", 250000, 4),
        ("PA", 750000, 5),
        ("fatalities", 0, 1),
        ("fatalities", 5, 2),
        ("fatalities", 26, 3),
        ("fatalities", 500, 5),
    ],
)
def test_bucket_index_from_value(metric: str, value: float, expected: int) -> None:
    assert scoring.bucket_index_from_value(metric, value) == expected


def test_multiclass_brier() -> None:
    probs = [0.1, 0.2, 0.3, 0.4]
    expected = sum((p - (1 if idx == 2 else 0)) ** 2 for idx, p in enumerate(probs))
    assert math.isclose(scoring.multiclass_brier(probs, 3), expected)


def test_log_score() -> None:
    probs = [0.1, 0.2, 0.3, 0.4]
    expected = -math.log(probs[3])
    assert math.isclose(scoring.log_score(probs, 4), expected)


def test_normalize_probs_handles_zero_total() -> None:
    assert scoring.normalize_probs([0, 0, 0]) == [pytest.approx(1 / 3)] * 3

