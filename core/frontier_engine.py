from __future__ import annotations

import numpy as np
import pandas as pd

from typing import List

from core.optimizer import PortfolioOptimizer
from models.portfolio_models import EfficientFrontierPoint


class FrontierEngine:
    @staticmethod
    def generate_efficient_frontier(
        annual_returns: pd.Series,
        covariance: pd.DataFrame,
        n_points: int = 50,
        min_return: float | None = None,
        max_return: float | None = None,
        min_weight: float = 0.01,
        max_weight: float = 0.25,
        sector_caps: dict[str, float] | None = None,
    ) -> List[EfficientFrontierPoint]:
        optimizer = PortfolioOptimizer(
            annual_returns,
            covariance,
            min_weight=min_weight,
            max_weight=max_weight,
            sector_caps=sector_caps,
        )
        if min_return is None:
            min_return = float(np.min(annual_returns))
        if max_return is None:
            max_return = float(np.max(annual_returns))
        target_returns = np.linspace(min_return, max_return, n_points)
        points: list[EfficientFrontierPoint] = []
        for target in target_returns:
            result = optimizer.optimize_target_return(target)
            if not result.success:
                continue
            volatility = float(np.sqrt(result.weights.T @ covariance.values @ result.weights))
            return_value = float(np.dot(result.weights, annual_returns.values))
            sharpe = optimizer._objective_neg_sharpe(result.weights) * -1
            points.append(
                EfficientFrontierPoint(
                    return_value=return_value,
                    volatility=volatility,
                    sharpe_ratio=sharpe,
                    weights={ticker: float(w) for ticker, w in zip(annual_returns.index, result.weights)},
                )
            )
        return points
