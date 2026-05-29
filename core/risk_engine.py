from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Optional

from utils.config import RISK_FREE_RATE


class RiskEngine:
    @staticmethod
    def volatility(returns: pd.Series) -> float:
        return float(returns.std(ddof=0) * np.sqrt(252))

    @staticmethod
    def portfolio_volatility(weights: np.ndarray, covariance: pd.DataFrame) -> float:
        return float(np.sqrt(weights.T @ covariance.values @ weights))

    @staticmethod
    def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float = RISK_FREE_RATE) -> float:
        if volatility <= 0:
            return 0.0
        return float((expected_return - risk_free_rate) / volatility)

    @staticmethod
    def sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
        downside = returns[returns < target_return] - target_return
        downside_std = float(np.sqrt(np.mean(np.square(downside))))
        if downside_std <= 0:
            return 0.0
        annual_return = float(returns.mean() * 252)
        return float((annual_return - target_return) / downside_std)

    @staticmethod
    def maximum_drawdown(price_series: pd.Series) -> float:
        cumulative = price_series.dropna().pct_change().add(1).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = cumulative / rolling_max - 1
        return float(drawdowns.min())

    @staticmethod
    def var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        if returns.empty:
            return 0.0
        return float(np.percentile(returns.dropna(), confidence_level * 100))

    @staticmethod
    def cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        if returns.empty:
            return 0.0
        var_threshold = RiskEngine.var(returns, confidence_level)
        tail_losses = returns[returns <= var_threshold]
        if tail_losses.empty:
            return float(var_threshold)
        return float(tail_losses.mean())
