from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AllocationItem(BaseModel):
    ticker: str
    name: str
    weight: float
    target_amount: Optional[float] = None
    shares: Optional[int] = None


class PortfolioMetrics(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    diversification_score: float
    concentration_score: float = 0.0
    effective_number_of_assets: float = 0.0
    diversification_ratio: float = 0.0
    correlation_exposure: float = 0.0
    exposure_balance_score: float = 0.0


class PortfolioResult(BaseModel):
    name: str
    allocations: Dict[str, float]
    metrics: PortfolioMetrics
    asset_names: Dict[str, str] = Field(default_factory=dict)
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    timestamp: Optional[date] = None


class EfficientFrontierPoint(BaseModel):
    return_value: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


class AgentResponse(BaseModel):
    content: str
    structured: Optional[PortfolioResult] = None
    explanation: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
