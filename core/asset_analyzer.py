from __future__ import annotations

import numpy as np
import pandas as pd


class AssetAnalyzer:
    @staticmethod
    def liquidity_score(prices: pd.DataFrame, min_observations: int = 252) -> pd.Series:
        history_ratio = prices.notna().sum() / prices.shape[0]
        return history_ratio

    @staticmethod
    def volatility_score(returns: pd.DataFrame) -> pd.Series:
        return returns.std(ddof=0) * np.sqrt(252)

    @staticmethod
    def diversification_score(weights: np.ndarray) -> float:
        return float(1.0 - np.sum(np.square(weights)))

    @staticmethod
    def filter_liquid_assets(prices: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
        observation_rate = prices.notna().mean()
        keepers = observation_rate[observation_rate >= threshold].index
        return prices[keepers]
