from __future__ import annotations

import numpy as np
import pandas as pd


class MetricsEngine:
    @staticmethod
    def cumulative_return(prices: pd.Series) -> pd.Series:
        return prices.pct_change().add(1).cumprod().fillna(1)

    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        return returns.rolling(window=window).std(ddof=0) * np.sqrt(252)

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 21) -> pd.Series:
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_volatility = MetricsEngine.rolling_volatility(returns, window)
        return rolling_return / rolling_volatility.replace(0, np.nan)

    @staticmethod
    def drawdown(price_series: pd.Series) -> pd.Series:
        cumulative = price_series.pct_change().add(1).cumprod()
        running_max = cumulative.cummax()
        return cumulative / running_max - 1
