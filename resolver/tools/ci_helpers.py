"""Helpers shared by GitHub Actions workflows."""

import calendar
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


def _coerce_to_istanbul(now: dt.datetime | None) -> dt.datetime:
    """Return ``now`` as an aware datetime in the Istanbul timezone."""

    if now is None:
        return dt.datetime.now(ISTANBUL_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=ISTANBUL_TZ)
    return now.astimezone(ISTANBUL_TZ)


def previous_month_istanbul(now: dt.datetime | None = None) -> str:
    """Return the previous calendar month in ``YYYY-MM`` using Istanbul time."""

    anchor = _coerce_to_istanbul(now)
    year = anchor.year
    month = anchor.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


@dataclass(frozen=True, slots=True)
class MonthlyWindow:
    """Container describing the previous month window in Istanbul time."""

    ym: str
    start_iso: str
    end_iso: str

    def to_env(self) -> dict[str, str]:
        """Return environment variables for snapshot workflows."""

        return {
            "SNAPSHOT_TARGET_YM": self.ym,
            "RESOLVER_PERIOD": self.ym,
            "RESOLVER_START_ISO": self.start_iso,
            "RESOLVER_END_ISO": self.end_iso,
        }


def monthly_snapshot_window(now: dt.datetime | None = None) -> MonthlyWindow:
    """Compute the snapshot window for the last completed month in Istanbul."""

    anchor = _coerce_to_istanbul(now)
    prev_ym = previous_month_istanbul(anchor)
    year, month = map(int, prev_ym.split("-"))
    start = dt.date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = dt.date(year, month, last_day)
    return MonthlyWindow(ym=prev_ym, start_iso=start.isoformat(), end_iso=end.isoformat())
