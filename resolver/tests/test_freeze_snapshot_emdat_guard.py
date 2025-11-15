import pandas as pd

from resolver.tools import freeze_snapshot


def test_run_validator_skips_for_acled_metrics(tmp_path, monkeypatch):
    facts_path = tmp_path / "facts_for_month.csv"
    df = pd.DataFrame(
        {
            "iso3": ["AFG", "ALB"],
            "ym": ["2025-11", "2025-11"],
            "as_of_date": ["2025-11-30", "2025-11-30"],
            "metric": ["events", "fatalities_battle_month"],
            "value": ["3", "6"],
        }
    )
    df.to_csv(facts_path, index=False)

    called = {"count": 0}

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        called["count"] += 1
        raise AssertionError("External validator should not be called for ACLED metrics")

    monkeypatch.setattr(freeze_snapshot.subprocess, "run", fake_run)

    freeze_snapshot.run_validator(facts_path)
    assert called["count"] == 0


def test_run_validator_calls_for_emdat_metrics(tmp_path, monkeypatch):
    facts_path = tmp_path / "facts_for_month.csv"
    df = pd.DataFrame(
        {
            "iso3": ["PHL"],
            "ym": ["2024-01"],
            "as_of_date": ["2024-01-31"],
            "metric": ["affected"],
            "value": ["1000"],
        }
    )
    df.to_csv(facts_path, index=False)

    called = {"count": 0}

    class DummyResult:
        def __init__(self):
            self.stdout = "Checked 1 rows; 0 issue(s) found."
            self.stderr = ""
            self.returncode = 0

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        called["count"] += 1
        return DummyResult()

    monkeypatch.setattr(freeze_snapshot.subprocess, "run", fake_run)

    freeze_snapshot.run_validator(facts_path)
    assert called["count"] == 1
