from __future__ import annotations

import datetime as dt

import pytest

from resolver.tools.load_and_derive import PeriodMonths


def test_period_months_from_quarter_label() -> None:
    period = PeriodMonths.from_label("2025Q4")
    assert period.label == "2025Q4"
    assert period.months == ("2025-10", "2025-11", "2025-12")


def test_period_months_alias_uses_current_quarter(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = dt.datetime(2024, 5, 15, tzinfo=dt.timezone.utc)

    class _Frozen(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:  # type: ignore[override]
            if tz is None:
                return fixed
            return fixed.astimezone(tz)

    monkeypatch.setattr("resolver.tools.load_and_derive.dt.datetime", _Frozen)

    period = PeriodMonths.from_label("ci-smoke")
    assert period.label == "2024Q2"
    assert period.months == ("2024-04", "2024-05", "2024-06")


def test_period_months_invalid_label() -> None:
    with pytest.raises(ValueError) as excinfo:
        PeriodMonths.from_label("foo")
    message = str(excinfo.value)
    assert "YYYYQ#" in message
    assert "ci" in message
