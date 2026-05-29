from __future__ import annotations

from pydantic import BaseModel


class AssetInfo(BaseModel):
    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
