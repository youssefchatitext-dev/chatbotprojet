from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from utils.config import COVARIANCE_METHOD


class CovarianceEngine:
    @staticmethod
    def compute_covariance(returns: pd.DataFrame, method: str = COVARIANCE_METHOD) -> pd.DataFrame:
        if method.lower() == "ledoit_wolf":
            try:
                lw = LedoitWolf()
                lw.fit(returns.dropna())
                covariance = pd.DataFrame(
                    lw.covariance_, index=returns.columns, columns=returns.columns
                )
                return covariance * 252
            except Exception:
                pass
        return returns.cov() * 252

    @staticmethod
    def compute_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
        diag = np.sqrt(np.diag(covariance))
        denom = np.outer(diag, diag)
        correlation = covariance / np.where(denom == 0, 1, denom)
        return pd.DataFrame(correlation, index=covariance.index, columns=covariance.columns)
