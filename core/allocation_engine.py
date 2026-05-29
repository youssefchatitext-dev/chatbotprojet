from __future__ import annotations

import math

from models.portfolio_models import PortfolioMetrics


class AllocationEngine:
    @staticmethod
    def score_portfolio(metrics: PortfolioMetrics, target_return: float | None, max_risk: float | None) -> float:
        score = 0.0
        if target_return is not None:
            return_diff = abs(metrics.expected_return - target_return)
            score += return_diff * 100
        if max_risk is not None:
            risk_diff = abs(metrics.volatility - max_risk)
            score += risk_diff * 100
        diversification_reward = max(0.0, metrics.diversification_score)
        score -= diversification_reward * 10
        return score

    @staticmethod
    def select_best_portfolio(
        candidates: list[PortfolioMetrics],
        target_return: float | None,
        max_risk: float | None,
    ) -> PortfolioMetrics | None:
        if not candidates:
            return None
        best = min(candidates, key=lambda metrics: AllocationEngine.score_portfolio(metrics, target_return, max_risk))
        return best

    @staticmethod
    def nearest_portfolio(
        metrics_list: list[PortfolioMetrics],
        target_return: float,
        max_risk: float,
    ) -> PortfolioMetrics | None:
        best_score = math.inf
        best = None
        for metrics in metrics_list:
            distance = math.sqrt((metrics.expected_return - target_return) ** 2 + (metrics.volatility - max_risk) ** 2)
            if distance < best_score:
                best_score = distance
                best = metrics
        return best
