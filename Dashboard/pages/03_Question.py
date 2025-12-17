# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import streamlit as st

from Dashboard.lib import api_client, components, state


def main() -> None:
    st.set_page_config(page_title="Question Drilldown", layout="wide")
    state.init_session_state()
    state.sync_query_params_from_url()

    st.title("Question Drilldown")
    st.caption("Inspect a question bundle with forecasts, scenarios, and transcripts.")

    components.render_global_filters()

    with st.form("question_form"):
        qid_value = st.text_input("Question ID", value=st.session_state.get("question_id", ""))
        question_url = st.text_input("Question URL (optional)", placeholder="https://www.metaculus.com/questions/12345/...")
        include_transcripts = st.checkbox(
            "Include transcripts", value=bool(st.session_state.get("include_transcripts", False))
        )
        submitted = st.form_submit_button("Load question")

    if submitted:
        parsed_qid = qid_value or components.parse_qid_from_url(question_url)
        if parsed_qid:
            st.session_state["question_id"] = str(parsed_qid)
            st.session_state["include_transcripts"] = include_transcripts
            state.update_query_params(question_id=parsed_qid)
        else:
            st.warning("Enter a question ID or a question URL to continue.")

    question_id = st.session_state.get("question_id")
    include_transcripts = bool(st.session_state.get("include_transcripts", False))

    if not question_id:
        st.info("Provide a question ID to load the bundle.")
        return

    params = {
        "question_id": question_id,
        "include_transcripts": include_transcripts,
    }

    try:
        bundle = api_client.api_get("/v1/question_bundle", params=params)
    except Exception as exc:  # pragma: no cover - UI error surface
        components.render_error("Unable to load question bundle", exc)
        return

    pipeline_tab, forecasts_tab, scenarios_tab, transcripts_tab = st.tabs(
        ["Pipeline Outputs", "Forecasts", "Scenarios", "Raw transcripts"]
    )

    with pipeline_tab:
        for label, key in (
            ("HS triage", "hs_triage"),
            ("HS country reports", "hs_country_reports"),
            ("Research", "research"),
        ):
            data = bundle.get(key)
            if data is not None:
                components.render_records_table(label, data, key=f"pipeline_{key}")

    with forecasts_tab:
        for label, key in (
            ("Ensemble forecasts", "ensemble_spd"),
            ("Per-model forecasts", "model_spds"),
            ("Resolutions", "resolutions"),
        ):
            data = bundle.get(key)
            if data is not None:
                components.render_records_table(label, data, key=f"forecast_{key}")

    with scenarios_tab:
        scenarios = bundle.get("hs_scenarios") or bundle.get("scenarios")
        if scenarios is not None:
            components.render_records_table("Scenarios", scenarios, key="scenarios_table")
        else:
            st.info("No scenarios returned for this question.")

    with transcripts_tab:
        if not include_transcripts:
            st.info("Enable 'Include transcripts' and reload to request transcripts.")
        else:
            transcripts = bundle.get("llm_calls")
            if transcripts is not None:
                components.render_records_table("LLM calls", transcripts, key="llm_calls")
            else:
                st.info("No transcripts returned for this question.")


if __name__ == "__main__":
    main()
