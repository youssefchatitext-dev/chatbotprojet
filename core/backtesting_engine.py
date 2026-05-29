from __future__ import annotations

import pandas as pd

from typing import Literal

from core.optimizer import PortfolioOptimizer
from core.portfolio_engine import PortfolioEngine
from utils.config import SUPPORTED_REBALANCE_FREQUENCIES


class BacktestingEngine:
    def __init__(self, rebalancing_frequency: Literal["monthly", "quarterly", "yearly"] = "monthly") -> None:
        if rebalancing_frequency not in SUPPORTED_REBALANCE_FREQUENCIES:
            raise ValueError(f"Unsupported rebalance frequency: {rebalancing_frequency}")
        self.frequency = rebalancing_frequency

    def rolling_optimization(
        self,
        prices: pd.DataFrame,
        target_return: float,
        max_risk: float,
        window_days: int = 252,
    ) -> list[dict[str, float]]:
        horizon = SUPPORTED_REBALANCE_FREQUENCIES[self.frequency]
        windows: list[dict[str, float]] = []
        for start in range(0, len(prices) - window_days, horizon):
            window_prices = prices.iloc[start : start + window_days]
            returns = window_prices.pct_change().dropna(how="all")
            if returns.shape[0] < window_days // 2:
                continue
            annual_returns = ((1 + returns.mean()) ** 252) - 1
            covariance = returns.cov() * 252
            optimizer = PortfolioOptimizer(annual_returns, covariance)
            result = optimizer.optimize_max_sharpe()
            if result.success:
                windows.append({ticker: float(weight) for ticker, weight in zip(returns.columns, result.weights)})
        return windows

    def rebalance_portfolio(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        turnover_limit: float = 0.2,
    ) -> dict[str, float]:
        output = current_weights.copy()
        for ticker, target_weight in target_weights.items():
            output[ticker] = target_weight
        total_turnover = sum(abs(output.get(t, 0.0) - current_weights.get(t, 0.0)) for t in output)
        if total_turnover > turnover_limit:
            scale = turnover_limit / total_turnover
            for ticker in output:
                output[ticker] = current_weights.get(ticker, 0.0) + (output[ticker] - current_weights.get(ticker, 0.0)) * scale
        total = sum(output.values())
        if total > 0:
            output = {ticker: weight / total for ticker, weight in output.items()}
        return output
