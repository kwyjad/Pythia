import pandas as pd
import yaml
from pathlib import Path

from resolver.ingestion.idmc import candidates as idmc_candidates
from resolver.ingestion.idmc.candidates import CANDIDATE_COLS, to_candidates_from_normalized
from tools.precedence_engine import apply_precedence


def test_precedence_selection_prefers_idmc(monkeypatch):
    monkeypatch.setattr(
        idmc_candidates, "_now_utc_date", lambda: pd.Timestamp("2024-04-30"), raising=False
    )

    idmc_normalized = pd.DataFrame(
        [
            {
                "iso3": "sdn",
                "as_of_date": "2024-03-31",
                "metric": "idp_displacement_new_idmc",
                "value": 1200,
                "series_semantics": "new",
                "source": "IDMC",
            }
        ]
    )
    idmc_candidates_df = to_candidates_from_normalized(idmc_normalized)

    dtm_candidate = pd.DataFrame(
        [
            {
                "iso3": "SDN",
                "as_of_date": pd.Timestamp("2024-03-31"),
                "metric": "internal_displacement_new",
                "value": 900,
                "source_system": "DTM",
                "collection_type": "flow_monitoring",
                "coverage": "national",
                "freshness_days": 30,
                "origin_iso3": None,
                "destination_iso3": None,
                "method_note": None,
                "series": None,
                "indicator": None,
                "indicator_kind": "explicit_flow",
                "qa_rank": 1,
            }
        ],
        columns=CANDIDATE_COLS,
    )

    candidates = pd.concat([idmc_candidates_df, dtm_candidate], ignore_index=True)

    config = yaml.safe_load(Path("tools/precedence_config.yml").read_text(encoding="utf-8"))
    selected = apply_precedence(candidates, config)

    assert not selected.empty
    conflict = selected[selected["metric"] == "conflict_onset1_pa"].iloc[0]
    assert conflict.value == 1200

    component_sources = conflict.component_sources
    assert isinstance(component_sources, list)
    assert component_sources
    first_component = component_sources[0]
    assert first_component["source_system"] == "IDMC"
    assert first_component["collection_type"] == "curated_event"
