from __future__ import annotations

import streamlit as st

from models.user_profile import UserProfile


def render_sidebar() -> tuple[UserProfile, float, str]:
    st.sidebar.title("Configuration du portefeuille")
    mode = st.sidebar.selectbox(
        "Mode",
        ["local", "ollama"],
        format_func=lambda x: "Local (hors ligne)" if x == "local" else "Ollama (conversational)",
    )
    risk_tolerance = st.sidebar.selectbox(
        "Tolérance au risque", ["low", "medium", "high"], index=1
    )
    investment_horizon = st.sidebar.slider(
        "Horizon d'investissement (années)", 1, 30, 5
    )
    objective = st.sidebar.selectbox(
        "Objectif", ["growth", "income", "balanced"], index=2
    )
    max_weight = st.sidebar.slider("Poids maximum par actif", 0.01, 0.5, 0.25, 0.01)
    min_weight = st.sidebar.slider("Poids minimum par actif", 0.0, 0.1, 0.01, 0.01)
    target_return = st.sidebar.number_input(
        "Rendement cible (%)", min_value=0.0, value=10.0, step=0.5
    )
    max_risk = st.sidebar.number_input(
        "Volatilité maximale (%)", min_value=0.0, value=18.0, step=0.5
    )
    capital = st.sidebar.number_input(
        "Capital à investir (MAD)", min_value=0.0, value=0.0, step=1000.0
    )
    user_profile = UserProfile(
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        income_preference=objective,
        growth_preference=objective,
        max_weight=max_weight,
        min_weight=min_weight,
        target_return=target_return / 100.0,
        max_risk=max_risk / 100.0,
    )
    return user_profile, capital, mode
