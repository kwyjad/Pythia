# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import streamlit as st

from Dashboard.lib import api_client, components, state


def main() -> None:
    st.set_page_config(page_title="Country Detail", layout="wide")
    state.init_session_state()
    state.sync_query_params_from_url()

    st.title("Country Detail")
    st.caption("Country-level questions, forecasts, outcomes, and HS summaries.")

    filters = components.render_global_filters()

    iso3 = st.text_input("ISO3 country code", value=st.session_state.get("iso3", ""))
    st.session_state["iso3"] = iso3.upper()

    if iso3:
        state.update_query_params(iso3=iso3)

    if not iso3:
        st.info("Enter an ISO3 code to load country details.")
        return

    params_common = {
        "iso3": iso3,
        "metric": filters.get("metric"),
        "target_month": filters.get("target_month"),
    }

    try:
        questions = api_client.api_get("/v1/questions", {"iso3": iso3, "latest_only": True})
        components.render_records_table("Questions", questions, key="questions_table")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load questions", exc)

    try:
        ensemble = api_client.api_get(
            "/v1/forecasts/ensemble",
            {**params_common, "horizon_m": filters.get("horizon_m")},
        )
        components.render_records_table("Forecast ensemble", ensemble, key="ensemble_table")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load ensemble forecasts", exc)

    try:
        outcomes = api_client.api_get(
            "/v1/resolutions",
            {**params_common, "horizon_m": filters.get("horizon_m")},
        )
        components.render_records_table("Outcomes", outcomes, key="outcomes_table")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load outcomes", exc)

    st.markdown("### Horizon Scanner")
    try:
        hs_runs = api_client.api_get("/v1/hs_runs", {"iso3": iso3})
        components.render_records_table("HS runs", hs_runs, key="hs_runs")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load HS runs", exc)

    try:
        hs_scenarios = api_client.api_get("/v1/hs_scenarios", {"iso3": iso3})
        components.render_records_table("HS scenarios", hs_scenarios, key="hs_scenarios")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load HS scenarios", exc)

    try:
        hs_country_reports = api_client.api_get("/v1/hs_country_reports", {"iso3": iso3})
        components.render_records_table("HS country reports", hs_country_reports, key="hs_reports")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load HS country reports", exc)


if __name__ == "__main__":
    main()
