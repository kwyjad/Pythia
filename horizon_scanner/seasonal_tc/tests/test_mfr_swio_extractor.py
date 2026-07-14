# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Fixture tests for the Météo-France La Réunion SWI seasonal outlook
extractor (no network). French fixtures mirror the phrasing observed across
the 2022-23 .. 2025-26 editions."""

from __future__ import annotations

from horizon_scanner.seasonal_tc import mfr_swio_scraper as mfr
from horizon_scanner.seasonal_tc.mfr_swio_scraper import (
    SeasonalForecast,
    build_candidate_urls,
    extract_swio_outlook,
    process_all,
)

# Phrasing based on the 2024-25 edition (with tercile probabilities).
FIXTURE_WITH_TERCILES = """
PRÉVISION SAISONNIÈRE D'ACTIVITÉ CYCLONIQUE DANS LE SUD-OUEST DE L'OCÉAN INDIEN — SAISON 2024-2025

Pour la saison cyclonique à venir, les modèles prévoient entre 9 et 13 systèmes
nommés, dont 4 à 7 pouvant atteindre le stade de cyclone tropical.

Les probabilités sont estimées à 50% de probabilité d'une activité supérieure à la
normale, 40% de probabilité d'une activité proche de la normale et 10% de
probabilité d'une activité inférieure à la normale.

Une faible tendance La Niña est attendue et le Dipôle de l'Océan Indien devrait
rester négatif au début de la saison.

Le canal du Mozambique et la côte est de Madagascar présentent un risque accru de
trajectoires menaçantes en seconde partie de saison.
"""

# Phrasing based on the 2025-26 edition (categorical only, no terciles).
FIXTURE_CATEGORICAL_ONLY = """
TENDANCE SAISONNIÈRE D'ACTIVITÉ CYCLONIQUE DANS LE SUD-OUEST DE L'OCÉAN INDIEN — SAISON 2025-2026

La tendance saisonnière privilégie une activité proche ou supérieure à la normale,
avec entre 9 et 14 systèmes nommés attendus sur l'ensemble du bassin.
""" + "\nremplissage " * 200


def test_extract_full_edition():
    f = extract_swio_outlook(FIXTURE_WITH_TERCILES, url="fixture://2024")
    assert f.season == "2024-25"
    assert f.season_year == 2024
    assert f.systems_range == "9-13"
    assert f.tc_stage_range == "4-7"
    assert f.prob_above == 0.5
    assert f.prob_near == 0.4
    assert f.prob_below == 0.1
    assert "La Niña" in f.enso_context or "La Nina" in f.enso_context
    assert "Mozambique" in f.regional_risk_note
    ctx = f.to_prompt_context()
    assert "50% above-normal" in ctx
    assert "9-13" in ctx
    assert "2024-25" in ctx


def test_extract_categorical_only_edition():
    f = extract_swio_outlook(FIXTURE_CATEGORICAL_ONLY, url="fixture://2025")
    assert f.season == "2025-26"
    assert f.systems_range == "9-14"
    assert f.categorical_outlook == "near to above normal"
    ctx = f.to_prompt_context()
    # No tercile line when the edition has no percentages.
    assert "above-normal," not in ctx
    assert "near to above normal" in ctx


def test_prompt_context_partial_terciles_does_not_raise():
    # The exact TSR-class crash shape: one tercile present, others None.
    f = SeasonalForecast(prob_above=0.5, prob_near=None, prob_below=None)
    ctx = f.to_prompt_context()  # must not raise
    assert "Season probabilities" not in ctx


def test_process_all_one_bad_url_keeps_batch(monkeypatch):
    """A failing candidate URL must not discard the batch (TSR-fix pattern)."""
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return FIXTURE_CATEGORICAL_ONLY

    monkeypatch.setattr(mfr, "fetch_page", fake_fetch)
    monkeypatch.setattr(mfr, "discover_outlook_url", lambda: None)
    forecasts = process_all(fetch_live=True, year=2025)
    assert len(forecasts) == 1
    assert forecasts[0].season == "2025-26"
    assert len(calls) >= 2


def test_process_all_no_source_returns_empty(monkeypatch):
    monkeypatch.setattr(mfr, "fetch_page", lambda url: (_ for _ in ()).throw(RuntimeError("404")))
    monkeypatch.setattr(mfr, "discover_outlook_url", lambda: None)
    assert process_all(fetch_live=True, year=2025) == []


def test_candidate_urls_cover_slug_variants():
    urls = build_candidate_urls(2025)
    joined = "\n".join(urls)
    assert "tendance-saisonniere" in joined
    assert "prevision-saisonniere" in joined
    assert "/fr/climat/" in joined and "/fr/actualites/" in joined
    assert "meteofrance.re" in joined and "meteofrance.yt" in joined
    # The unversioned 2023-style slug is present.
    assert any(u.endswith("locean-indien-saison") for u in urls)
