# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import streamlit as st

from Dashboard.lib import api_client, components, state


def main() -> None:
    st.set_page_config(page_title="Risk Index", layout="wide")
    state.init_session_state()
    state.sync_query_params_from_url()

    st.title("Risk Index")
    st.caption("Table of risk scores filtered by metric, horizon, and target month.")

    filters = components.render_global_filters()

    params = {
        "metric": filters.get("metric"),
        "horizon_m": filters.get("horizon_m"),
        "normalize": filters.get("normalize"),
    }
    if filters.get("target_month"):
        params["target_month"] = filters["target_month"]

    try:
        data = api_client.api_get("/v1/risk_index", params=params)
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load risk index", exc)
        return

    df = components.records_to_dataframe(data)
    if df is None or df.empty:
        st.info("No risk index data available.")
        return

    if "iso3" in df.columns:
        df["country_page"] = df["iso3"].apply(lambda iso: f"./02_Country?iso3={iso}")
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "country_page": st.column_config.LinkColumn(
                    "Country page", display_text=r"Country detail"
                )
            },
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
