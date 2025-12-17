# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import streamlit as st

from Dashboard.lib import api_client, components, state


def main() -> None:
    st.set_page_config(page_title="Pythia Dashboard", layout="wide")
    state.init_session_state()
    state.sync_query_params_from_url()

    st.title("Pythia Dashboard")
    st.caption("Multi-page, API-first dashboard using the /v1 endpoints.")

    components.render_global_filters()
    st.divider()

    st.subheader("Pages")
    components.render_navigation_links()

    st.markdown(
        """
        The dashboard is organized into focused pages:

        * **Runs & Health** – pipeline status, costs, and recent runs.
        * **Risk Index** – cross-country table filtered by metric, horizon, and target month.
        * **Country Detail** – country-level questions, forecasts, resolutions, and HS summaries.
        * **Question Drilldown** – full bundle for a specific question (pipeline outputs, forecasts, transcripts).
        """
    )

    with st.sidebar:
        st.header("Navigation")
        components.render_navigation_links(use_container=True)
        st.divider()
        st.caption(f"API base: {api_client.get_base_url()}")


if __name__ == "__main__":
    main()
