"""FabPilot Streamlit UI.

Run with ``fabpilot ui`` (see ``fabpilot.main``) or ``streamlit run fabpilot/ui/app.py``.

The UI talks to the *service* directly (same process) rather than the HTTP API,
so it works even with a ``fake`` LLM and an in-memory database.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from fabpilot.api.service import FabPilotService, FabPilotServiceError
from fabpilot.config import get_settings


@st.cache_resource
def get_service() -> FabPilotService:
    return FabPilotService(get_settings())


st.set_page_config(page_title="FabPilot", layout="wide")
st.title("FabPilot")
st.caption("Hypothesis-Driven Multi-Agent System for Industrial Time-Series Analysis & Fault Diagnosis")

service = get_service()
settings = service.settings

tab_data, tab_forecast, tab_anomaly, tab_knowledge, tab_task = st.tabs(
    ["Equipment / FDC", "Forecast", "Anomaly Detection", "Knowledge Base", "Agent Task"]
)

# ---- Equipment / FDC ------------------------------------------------------- #
with tab_data:
    st.header("Equipment")
    try:
        equipment = service.list_equipment()
    except FabPilotServiceError as exc:
        st.error(str(exc))
        equipment = []

    if equipment:
        codes = [e.get("code") or e.get("id") for e in equipment]
        selected = st.selectbox("Equipment", codes)
        eq = next((e for e in equipment if (e.get("code") or e.get("id")) == selected), None)
        if eq:
            st.json(eq)

        st.subheader("Sensor readings")
        channel = st.selectbox(
            "Channel",
            ["pressure", "temperature", "gas_flow", "rf_power", "vacuum_pressure", "bias_voltage", "helium_flow"],
        )
        limit = st.slider("Points", 50, 2000, 300)
        try:
            data = service.sensor_data(selected, channel=channel, limit=limit)
        except FabPilotServiceError as exc:
            st.error(str(exc))
        else:
            points = data.get("last_10_points") or []
            if points:
                st.line_chart({channel: [p.get(channel) for p in points]})
                st.caption(f"{data.get('n_points', len(points))} points total; showing last {len(points)}")
            else:
                st.info("No sensor data.")
    else:
        st.info("No equipment found. Run the seed script first (`fabpilot seed`).")

# ---- Forecast -------------------------------------------------------------- #
with tab_forecast:
    st.header("Forecast")
    col1, col2, col3 = st.columns(3)
    with col1:
        equipment_id = st.text_input("Equipment", value="ETCH-01")
    with col2:
        target = st.selectbox(
            "Target channel",
            ["gas_flow", "pressure", "temperature", "rf_power", "vacuum_pressure", "bias_voltage", "helium_flow"],
        )
    with col3:
        model = st.selectbox("Model", ["statistical_baseline", "patchtst", "itransformer"])
    if st.button("Run forecast"):
        try:
            result = service.forecast(equipment_id, target=target, model=model)
        except FabPilotServiceError as exc:
            st.error(str(exc))
        else:
            forecast = result.get("forecast") or result.get("points") or []
            if forecast:
                st.line_chart({target: [p.get("value") for p in forecast]})
            st.json(result)

# ---- Anomaly Detection ----------------------------------------------------- #
with tab_anomaly:
    st.header("Anomaly Detection")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        equipment_id = st.text_input("Equipment", value="ETCH-01", key="anom_eq")
    with col2:
        channel = st.selectbox("Channel", ["gas_flow", "pressure", "temperature", "rf_power"], key="anom_ch")
    with col3:
        window = st.number_input("Window", min_value=2, value=24)
    with col4:
        threshold = st.number_input("Threshold", min_value=0.5, value=3.0)
    if st.button("Detect anomalies"):
        try:
            result = service.detect_anomalies(equipment_id, channel=channel, window=int(window), threshold=float(threshold))
        except FabPilotServiceError as exc:
            st.error(str(exc))
        else:
            anomalies = result.get("anomalies") or []
            st.metric("Anomalies", len(anomalies))
            st.json(result)

# ---- Knowledge Base -------------------------------------------------------- #
with tab_knowledge:
    st.header("Knowledge Base")
    query = st.text_input("Query", value="gas flow drift")
    k = st.slider("Results", 1, 20, 5)
    if st.button("Search"):
        try:
            result = service.search_knowledge(query, k=k)
        except FabPilotServiceError as exc:
            st.error(str(exc))
        else:
            docs = result.get("results") or []
            for d in docs:
                with st.expander(f"{d.get('source', d.get('title', ''))}  (score {d.get('score', 0):.3f})"):
                    st.markdown(d.get("text", ""))

# ---- Agent Task ------------------------------------------------------------ #
with tab_task:
    st.header("Run agent task")
    st.caption("Routes to the supervisor + specialised agents via the hypothesis-evidence loop.")
    user_task = st.text_area(
        "Task",
        value="ETCH-01 的 gas_flow 最近出现漂移，请诊断根因并给出处置建议。",
        height=120,
    )
    if st.button("Run task", type="primary"):
        with st.spinner("Running agent system..."):
            try:
                state = service.run_task(user_task)
            except FabPilotServiceError as exc:
                st.error(str(exc))
            else:
                st.subheader("Status")
                st.write(state.get("status"))
                if state.get("delegation_records"):
                    st.subheader("Delegated agents")
                    st.write([r.get("agent") for r in state["delegation_records"]])
                if state.get("hypotheses"):
                    st.subheader("Hypotheses")
                    st.json(state["hypotheses"])
                st.subheader("Final report")
                st.markdown(state.get("final_report", ""))
