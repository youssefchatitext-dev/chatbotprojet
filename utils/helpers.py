from __future__ import annotations

import numpy as np
import pandas as pd
import random

from .config import RANDOM_SEED


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def geometric_annual_returns(returns: pd.DataFrame) -> pd.Series:
    return ((1 + returns.mean()) ** 252) - 1


def quantile_series(series: pd.Series, quantile: float) -> float:
    return float(series.quantile(quantile))


def cumulative_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().add(1).cumprod().fillna(1)


def format_percentage(value: float, precision: int = 2) -> str:
    return f"{value * 100:.{precision}f}%" if abs(value) < 10 else f"{value:.{precision}f}"
