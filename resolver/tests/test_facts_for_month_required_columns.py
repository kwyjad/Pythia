import pandas as pd

from resolver.tools import freeze_snapshot


def test_preview_adds_required_columns(tmp_path):
    facts_path = tmp_path / "facts_for_month.csv"
    pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "ym": "2024-01",
                "event_id": "evt-001",
                "metric": "affected",
                "value": "100",
            },
            {
                "iso3": "BBB",
                "ym": "2024-01",
                "event_id": "evt-002",
                "metric": "in_need",
                "value": "50",
            },
        ]
    ).to_csv(facts_path, index=False)

    freeze_snapshot._normalize_facts_for_validation(facts_path)  # type: ignore[attr-defined]

    frame = pd.read_csv(facts_path)
    for column in ["hazard_code", "hazard_label", "hazard_class", "as_of_date"]:
        assert column in frame.columns
        assert frame[column].astype(str).str.strip().ne("").all()

    assert frame["hazard_code"].eq("UNK").all()
    assert frame["hazard_label"].eq("Unknown").all()
    assert frame["hazard_class"].eq("unknown").all()
    assert frame["as_of_date"].eq("2024-01-31").all()
