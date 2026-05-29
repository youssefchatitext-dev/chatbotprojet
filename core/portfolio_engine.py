from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Optional

from core.risk_engine import RiskEngine
from models.portfolio_models import PortfolioMetrics, PortfolioResult
from utils.helpers import geometric_annual_returns


class PortfolioEngine:
    def __init__(self, risk_free_rate: float = 0.03) -> None:
        self.risk_free_rate = risk_free_rate

    def build_portfolio(
        self,
        name: str,
        weights: np.ndarray,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        asset_labels: dict[str, str] | None = None,
        price_history: pd.DataFrame | None = None,
        target_return: float | None = None,
        max_risk: float | None = None,
    ) -> PortfolioResult:
        annual_returns = geometric_annual_returns(returns)
        expected_return = float(np.dot(weights, annual_returns.values))
        volatility = RiskEngine.portfolio_volatility(weights, covariance)
        real_returns = returns.dot(weights)
        sortino_ratio = RiskEngine.sortino_ratio(real_returns)
        max_dd = 0.0
        if price_history is not None:
            portfolio_prices = price_history.dot(weights)
            max_dd = abs(RiskEngine.maximum_drawdown(portfolio_prices))
        concentration = float(np.sum(np.square(weights)))
        diversification = float(1.0 - concentration)
        effective_assets = float(1.0 / concentration) if concentration > 0 else 0.0
        asset_volatility = np.sqrt(np.diag(covariance.values))
        weighted_asset_volatility = float(np.dot(weights, asset_volatility))
        diversification_ratio = float(weighted_asset_volatility / volatility) if volatility > 0 else 0.0
        corr = returns.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        weight_outer = np.outer(weights, weights)
        off_diagonal = ~np.eye(len(weights), dtype=bool)
        correlation_exposure = float(np.sum(np.abs(corr[off_diagonal]) * weight_outer[off_diagonal]))
        max_weight = float(np.max(weights)) if len(weights) else 0.0
        exposure_balance = float(max(0.0, 1.0 - max_weight))
        var_95 = RiskEngine.var(real_returns)
        cvar_95 = RiskEngine.cvar(real_returns)

        metrics = PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=RiskEngine.sharpe_ratio(expected_return, volatility, self.risk_free_rate),
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            diversification_score=diversification,
            concentration_score=concentration,
            effective_number_of_assets=effective_assets,
            diversification_ratio=diversification_ratio,
            correlation_exposure=correlation_exposure,
            exposure_balance_score=exposure_balance,
        )

        allocations = {
            ticker: float(weight)
            for ticker, weight in zip(returns.columns, weights)
            if float(weight) > 0
        }
        asset_names = asset_labels or {}

        return PortfolioResult(
            name=name,
            allocations=allocations,
            metrics=metrics,
            asset_names=asset_names,
            target_return=target_return,
            max_risk=max_risk,
        )
