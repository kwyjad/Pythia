# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import streamlit as st

from Dashboard.lib import api_client, components, state


def main() -> None:
    st.set_page_config(page_title="Runs & Health", layout="wide")
    state.init_session_state()
    state.sync_query_params_from_url()

    st.title("Runs & Health")
    st.caption("Diagnostics, pipeline controls, and recent run history.")

    components.render_global_filters()

    col1, col2 = st.columns(2)

    with col1:
        try:
            diagnostics = api_client.api_get("/v1/diagnostics/summary")
            components.render_key_value_pairs("Health summary", diagnostics)
        except Exception as exc:  # pragma: no cover - UI error surface
            components.render_error("Unable to load health summary", exc)

    with col2:
        try:
            costs = api_client.api_get("/v1/llm/costs/summary")
            components.render_records_table("LLM cost summary", costs, key="costs_table")
        except Exception as exc:  # pragma: no cover - UI error surface
            components.render_error("Unable to load cost summary", exc)

    st.markdown("### Run pipeline")
    st.caption("POST /v1/run")
    if st.button("Run pipeline", type="primary"):
        try:
            response = api_client.api_post("/v1/run", {})
            st.success("Pipeline run triggered.")
            st.json(response)
        except Exception as exc:  # pragma: no cover - UI error surface
            components.render_error("Unable to trigger pipeline", exc)

    st.divider()

    try:
        runs = api_client.api_get("/v1/ui_runs")
        components.render_records_table("Recent runs", runs, key="runs_table")
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load recent runs", exc)


if __name__ == "__main__":
    main()
