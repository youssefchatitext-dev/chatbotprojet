from __future__ import annotations

import streamlit as st

from models.portfolio_models import PortfolioResult


def render_portfolio_summary(portfolio: PortfolioResult) -> None:
    st.subheader("Résumé du portefeuille")
    st.metric("Rendement attendu", f"{portfolio.metrics.expected_return * 100:.2f}%")
    st.metric("Volatilité", f"{portfolio.metrics.volatility * 100:.2f}%")
    st.metric("Sharpe", f"{portfolio.metrics.sharpe_ratio:.2f}")
    st.metric("Sortino", f"{portfolio.metrics.sortino_ratio:.2f}")
    st.metric("Max Drawdown", f"{portfolio.metrics.max_drawdown * 100:.2f}%")


def render_chat_history(messages: list[dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
