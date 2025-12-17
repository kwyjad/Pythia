# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

from Dashboard.lib import state


def render_navigation_links(use_container: bool = False) -> None:
    links: List[Tuple[str, str, str]] = [
        ("Dashboard/pages/00_Runs_Health.py", "Runs & Health", "🩺"),
        ("Dashboard/pages/01_Risk_Index.py", "Risk Index", "📈"),
        ("Dashboard/pages/02_Country.py", "Country Detail", "🗺️"),
        ("Dashboard/pages/03_Question.py", "Question Drilldown", "❓"),
    ]

    container = st.container() if use_container else st
    if hasattr(container, "page_link"):
        for page, label, icon in links:
            container.page_link(page, label=label, icon=icon)
    else:  # pragma: no cover - Streamlit fallback
        bullets = "\n".join(f"- {icon} [{label}]({page})" for page, label, icon in links)
        container.markdown(bullets)


def render_global_filters() -> Dict[str, Any]:
    state.init_session_state()

    with st.container():
        st.subheader("Global filters")
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1, 1])
        metric_options = ["PA", "HS", "ID"]
        metric_value = str(st.session_state.get("metric", "PA"))
        metric_index = metric_options.index(metric_value) if metric_value in metric_options else 0

        with col1:
            st.session_state["metric"] = st.selectbox(
                "Metric",
                options=metric_options,
                index=metric_index,
                help="Metric used across risk index and forecasts.",
            )
        with col2:
            st.session_state["target_month"] = st.text_input(
                "Target month",
                value=str(st.session_state.get("target_month", "")),
                placeholder="YYYY-MM (blank = latest)",
                help="Leave blank to use the latest month returned by the API.",
            )
        with col3:
            st.session_state["horizon_m"] = st.number_input(
                "Horizon (months)",
                min_value=0,
                max_value=24,
                value=int(st.session_state.get("horizon_m", 1) or 0),
                step=1,
            )
        with col4:
            st.session_state["normalize"] = st.checkbox(
                "Normalize",
                value=bool(st.session_state.get("normalize", True)),
                help="If checked, show normalized scores where supported.",
            )

    return state.get_filters()


def render_records_table(title: str, records: Any, *, key: str | None = None) -> None:
    st.markdown(f"### {title}")
    df = records_to_dataframe(records)
    if df is None or df.empty:
        st.info("No data available for this section.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True, key=key)


def render_key_value_pairs(title: str, data: Dict[str, Any] | None) -> None:
    st.markdown(f"### {title}")
    if not data:
        st.info("No data available.")
        return

    df = pd.DataFrame(data.items(), columns=["key", "value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def records_to_dataframe(records: Any) -> pd.DataFrame | None:
    if records is None:
        return None
    if isinstance(records, list):
        if not records:
            return None
        return pd.DataFrame.from_records(records)
    if isinstance(records, dict):
        if all(isinstance(v, (list, tuple, dict)) for v in records.values()):
            return pd.DataFrame.from_dict(records, orient="index").reset_index(names=["key"])
        return pd.DataFrame(records.items(), columns=["key", "value"])
    return pd.DataFrame([{"value": records}])


def parse_qid_from_url(url: str) -> str | None:
    match = re.search(r"/questions/(\d+)/", url)
    return match.group(1) if match else None


def render_error(message: str, exc: Exception | None = None) -> None:
    if exc:
        st.error(f"{message}: {exc}")
    else:
        st.error(message)
