from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.ci import compute_backfill_window


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2024, 5, 17, tzinfo=tz)


def test_compute_backfill_window_env_outputs(tmp_path: Path, monkeypatch) -> None:
    output_file = tmp_path / "github_output"
    env_file = tmp_path / "github_env"

    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("MONTHS_INPUT", "3")
    monkeypatch.setenv("TIMEZONE", "UTC")
    monkeypatch.setattr(compute_backfill_window, "datetime", _FixedDatetime)

    rc = compute_backfill_window.main()
    assert rc == 0

    output_data = {
        key: value
        for key, value in (
            line.split("=", 1) for line in output_file.read_text().strip().splitlines()
        )
    }
    assert output_data["months"] == "2024-03,2024-04,2024-05"
    assert output_data["start_iso"] == "2024-03-01"
    assert output_data["end_iso"] == "2024-05-31"
    assert output_data["month_count"] == "3"

    env_data = {
        key: value
        for key, value in (
            line.split("=", 1)
            for line in env_file.read_text().strip().splitlines()
        )
    }
    assert env_data["BACKFILL_MONTHS"] == "2024-03 2024-04 2024-05"
    assert env_data["BACKFILL_MONTHS_CSV"] == "2024-03,2024-04,2024-05"
    assert env_data["BACKFILL_START_ISO"] == "2024-03-01"
    assert env_data["BACKFILL_END_ISO"] == "2024-05-31"
