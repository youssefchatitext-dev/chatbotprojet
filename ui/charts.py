from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def allocation_pie_chart(allocations: dict[str, float]) -> go.Figure:
    allocation_df = pd.DataFrame(
        {
            "Asset": list(allocations.keys()),
            "Weight": [weight * 100 for weight in allocations.values()],
        }
    )
    return px.pie(allocation_df, names="Asset", values="Weight", title="Répartition du portefeuille")


def cumulative_returns_chart(price_series: pd.Series) -> go.Figure:
    cum = price_series.pct_change().add(1).cumprod().fillna(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative Return"))
    fig.update_layout(title="Performance cumulée", xaxis_title="Date", yaxis_title="Cumul des retours")
    return fig


def drawdown_chart(drawdown_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values, mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def rolling_sharpe_chart(rolling_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name="Rolling Sharpe"))
    fig.update_layout(title="Sharpe rolling", xaxis_title="Date", yaxis_title="Sharpe ratio")
    return fig


def efficient_frontier_chart(frontier_df: pd.DataFrame) -> go.Figure:
    fig = px.line(frontier_df, x="volatility", y="return", title="Efficient Frontier")
    fig.update_traces(mode="markers+lines")
    fig.update_layout(xaxis_title="Volatilité", yaxis_title="Rendement attendu")
    return fig
