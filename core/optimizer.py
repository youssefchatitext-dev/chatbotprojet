from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.optimize import minimize

from utils.config import MAX_WEIGHT, MIN_WEIGHT, OPTIMIZATION_METHOD, DIVERSIFICATION_PENALTY, RISK_PARITY_PENALTY
from core.risk_engine import RiskEngine


@dataclass
class OptimizationResult:
    weights: np.ndarray
    success: bool
    message: str


class PortfolioOptimizer:
    def __init__(
        self,
        annual_returns: pd.Series,
        covariance: pd.DataFrame,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
        sector_caps: dict[str, float] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.annual_returns = annual_returns
        self.covariance = covariance
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sector_caps = sector_caps or {}
        self.sector_map = sector_map or {}
        self.num_assets = len(annual_returns)
        self.initial_weights = np.repeat(1.0 / self.num_assets, self.num_assets)

    def _sector_mask(self, sector: str) -> np.ndarray:
        if not self.sector_map:
            return np.zeros(self.num_assets)
        return np.array(
            [1.0 if self.sector_map.get(ticker) == sector else 0.0 for ticker in self.annual_returns.index],
            dtype=float,
        )

    def _bounds(self) -> list[tuple[float, float]]:
        return [(self.min_weight, self.max_weight) for _ in range(self.num_assets)]

    def _sum_to_one_constraint(self) -> dict:
        return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    def _target_return_constraint(self, target_return: float) -> dict:
        return {
            "type": "eq",
            "fun": lambda weights: float(np.dot(weights, self.annual_returns.values) - target_return),
        }

    def _sector_constraints(self) -> list[dict]:
        constraints: list[dict] = []
        if not self.sector_caps:
            return constraints
        for sector, cap in self.sector_caps.items():
            def make_constraint(sector: str, cap: float) -> dict:
                return {
                    "type": "ineq",
                    "fun": lambda weights, sector=sector, cap=cap: cap - np.sum(
                        weights * self._sector_mask(sector)
                    ),
                }
            constraints.append(make_constraint(sector, cap))
        return constraints

    @staticmethod
    def _herfindahl_hirschman(weights: np.ndarray) -> float:
        return float(np.sum(np.square(weights)))

    def _objective_variance(self, weights: np.ndarray) -> float:
        return float(weights.T @ self.covariance.values @ weights)

    def _objective_minimum_volatility(self, weights: np.ndarray) -> float:
        return self._objective_variance(weights)

    def _objective_neg_sharpe(self, weights: np.ndarray) -> float:
        volatility = RiskEngine.portfolio_volatility(weights, self.covariance)
        expected_return = float(np.dot(weights, self.annual_returns.values))
        sharpe = RiskEngine.sharpe_ratio(expected_return, volatility)
        concentration = self._herfindahl_hirschman(weights)
        return float(-sharpe + DIVERSIFICATION_PENALTY * concentration)

    def _objective_risk_parity(self, weights: np.ndarray) -> float:
        portfolio_variance = float(weights.T @ self.covariance.values @ weights)
        marginal_contribution = self.covariance.values @ weights
        risk_contributions = weights * marginal_contribution
        target = portfolio_variance / self.num_assets
        penalty = DIVERSIFICATION_PENALTY * self._herfindahl_hirschman(weights)
        return float(np.sum((risk_contributions - target) ** 2) + penalty)

    def _objective_diversification(self, weights: np.ndarray) -> float:
        return float(self._herfindahl_hirschman(weights))

    def _objective_balanced(self, weights: np.ndarray) -> float:
        expected_return = float(np.dot(weights, self.annual_returns.values))
        volatility = RiskEngine.portfolio_volatility(weights, self.covariance)
        sharpe = RiskEngine.sharpe_ratio(expected_return, volatility)
        concentration = self._herfindahl_hirschman(weights)
        return float(-0.85 * sharpe + 0.15 * concentration + 0.01 * volatility)

    def _solve(self, objective, constraints: list[dict]) -> OptimizationResult:
        result = minimize(
            objective,
            self.initial_weights,
            method=OPTIMIZATION_METHOD,
            bounds=self._bounds(),
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False, "maxiter": 1000},
        )
        if result.success:
            weights = np.maximum(result.x, 0.0)
            weights = weights / np.sum(weights)
            return OptimizationResult(weights=weights, success=True, message=result.message)
        return OptimizationResult(weights=self.initial_weights, success=False, message=result.message)

    def optimize_max_sharpe(self) -> OptimizationResult:
        constraints = [self._sum_to_one_constraint()] + self._sector_constraints()
        return self._solve(self._objective_neg_sharpe, constraints)

    def optimize_minimum_variance(self) -> OptimizationResult:
        constraints = [self._sum_to_one_constraint()] + self._sector_constraints()
        return self._solve(self._objective_variance, constraints)

    def optimize_minimum_volatility(self) -> OptimizationResult:
        return self.optimize_minimum_variance()

    def _target_return_lower_bound_constraint(self, target_return: float) -> dict:
        return {
            "type": "ineq",
            "fun": lambda weights: float(np.dot(weights, self.annual_returns.values) - target_return),
        }

    def optimize_minimum_variance_for_return(self, target_return: float) -> OptimizationResult:
        constraints = [
            self._sum_to_one_constraint(),
            self._target_return_lower_bound_constraint(target_return),
        ] + self._sector_constraints()
        return self._solve(self._objective_variance, constraints)

    def optimize_target_return(self, target_return: float) -> OptimizationResult:
        constraints = [
            self._sum_to_one_constraint(),
            self._target_return_constraint(target_return),
        ] + self._sector_constraints()
        return self._solve(self._objective_variance, constraints)

    def optimize_risk_parity(self) -> OptimizationResult:
        constraints = [self._sum_to_one_constraint()] + self._sector_constraints()
        return self._solve(self._objective_risk_parity, constraints)

    def optimize_diversification(self) -> OptimizationResult:
        constraints = [self._sum_to_one_constraint()] + self._sector_constraints()
        return self._solve(self._objective_diversification, constraints)

    def optimize_balanced(self) -> OptimizationResult:
        constraints = [self._sum_to_one_constraint()] + self._sector_constraints()
        return self._solve(self._objective_balanced, constraints)

    def sample_random_portfolios(self, num_portfolios: int = 20) -> list[np.ndarray]:
        rng = np.random.default_rng(42)
        portfolios = []
        for _ in range(num_portfolios):
            weights = rng.random(self.num_assets)
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights /= np.sum(weights)
            portfolios.append(weights)
        return portfolios

    def equal_weight(self) -> OptimizationResult:
        weights = np.repeat(1.0 / self.num_assets, self.num_assets)
        return OptimizationResult(weights=weights, success=True, message="equal weight benchmark")
