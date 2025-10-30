import pandas as pd

from resolver.ingestion.idmc import candidates as idmc_candidates
from resolver.ingestion.idmc.candidates import CANDIDATE_COLS, to_candidates_from_normalized


def test_to_candidates_from_normalized(monkeypatch):
    monkeypatch.setattr(
        idmc_candidates, "_now_utc_date", lambda: pd.Timestamp("2024-04-30"), raising=False
    )

    df_norm = pd.DataFrame(
        [
            {
                "iso3": "nga",
                "as_of_date": "2024-03-31",
                "metric": "idp_displacement_new_idmc",
                "value": "1500",
                "series_semantics": "new",
                "source": "IDMC",
            },
            {
                "iso3": " ",
                "as_of_date": "invalid",
                "metric": "idp_displacement_new_idmc",
                "value": "bad",
            },
        ]
    )

    result = to_candidates_from_normalized(df_norm)

    assert list(result.columns) == CANDIDATE_COLS
    assert len(result) == 1

    row = result.iloc[0]
    assert row.iso3 == "NGA"
    assert row.metric == "internal_displacement_new"
    assert row.series == "IDU"
    assert row.indicator_kind == "explicit_flow"
    assert row.source_system == "IDMC"
    assert row.collection_type == "curated_event"
    assert row.coverage == "national"
    assert row.freshness_days == 30
    assert pd.isna(row.origin_iso3)
    assert pd.isna(row.destination_iso3)
    assert row.method_note.startswith("IDU preliminary")
    assert row.qa_rank == 3


def test_to_candidates_from_normalized_empty():
    empty = pd.DataFrame(
        columns=["iso3", "as_of_date", "metric", "value", "series_semantics", "source"]
    )
    result = to_candidates_from_normalized(empty)
    assert list(result.columns) == CANDIDATE_COLS
    assert result.empty
