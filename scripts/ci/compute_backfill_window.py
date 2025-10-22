from __future__ import annotations

import calendar
import os
import sys
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python < 3.9 on some runners
    ZoneInfo = None  # type: ignore[assignment]


def main() -> int:
    months_back = int(os.environ.get("MONTHS_INPUT", "12") or "12")
    tz_name = os.environ.get("TIMEZONE", "UTC")
    now = datetime.now(ZoneInfo(tz_name) if ZoneInfo else None)

    year, month = now.year, now.month
    months: list[str] = []
    for _ in range(months_back):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    months.reverse()
    if not months:
        print("months_back must be >= 1", file=sys.stderr)
        return 1

    start_iso = months[0] + "-01"
    end_year, end_month = map(int, months[-1].split("-"))
    last_day = calendar.monthrange(end_year, end_month)[1]
    end_iso = f"{end_year:04d}-{end_month:02d}-{last_day:02d}"

    outputs = {
        "months": ",".join(months),
        "start_iso": start_iso,
        "end_iso": end_iso,
        "month_count": str(len(months)),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                print(f"{key}={value}", file=handle)

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"BACKFILL_MONTHS={' '.join(months)}\n")
            handle.write(f"BACKFILL_MONTHS_CSV={outputs['months']}\n")
            handle.write(f"BACKFILL_START_ISO={start_iso}\n")
            handle.write(f"BACKFILL_END_ISO={end_iso}\n")

    print(f"Derived months: {outputs['months']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
