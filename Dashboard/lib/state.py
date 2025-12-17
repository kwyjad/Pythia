# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

DEFAULT_FILTERS: Dict[str, Any] = {
    "metric": "PA",
    "target_month": "",
    "horizon_m": 1,
    "normalize": True,
}

SESSION_DEFAULTS: Dict[str, Any] = {
    **DEFAULT_FILTERS,
    "iso3": "",
    "question_id": "",
    "include_transcripts": False,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def get_filters() -> Dict[str, Any]:
    return {key: st.session_state.get(key, DEFAULT_FILTERS[key]) for key in DEFAULT_FILTERS}


def sync_query_params_from_url() -> None:
    params = st.experimental_get_query_params()
    iso3 = params.get("iso3", [])
    if iso3:
        st.session_state["iso3"] = iso3[0].upper()

    qid = params.get("question_id", []) or params.get("qid", [])
    if qid:
        st.session_state["question_id"] = str(qid[0])

    target_month = params.get("target_month", [])
    if target_month:
        st.session_state["target_month"] = target_month[0]


def update_query_params(**kwargs: Any) -> None:
    params = st.experimental_get_query_params()
    for key, value in kwargs.items():
        if value in (None, ""):
            params.pop(key, None)
        else:
            params[key] = [str(value)]
    st.experimental_set_query_params(**params)
