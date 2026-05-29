from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class UserProfile(BaseModel):
    risk_tolerance: Optional[str] = Field(default="medium")
    investment_horizon: Optional[int] = Field(default=5, ge=1)
    income_preference: Optional[str] = Field(default="balanced")
    growth_preference: Optional[str] = Field(default="balanced")
    preferred_sectors: List[str] = Field(default_factory=list)
    max_weight: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)
    min_weight: Optional[float] = Field(default=0.01, ge=0.0, le=1.0)
    target_return: Optional[float] = None
    max_risk: Optional[float] = None

    @validator("risk_tolerance")
    def normalize_risk_tolerance(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return "medium"
        return value.lower()

    @validator("income_preference", "growth_preference")
    def normalize_preferences(cls, value: Optional[str]) -> Optional[str]:
        return value.lower() if value else value
