from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import INFO_CSV, RAW_STOCK_DIR
from utils.validators import validate_price_matrix


class DataService:
    def __init__(self, stock_dir: Path = RAW_STOCK_DIR, info_csv: Path = INFO_CSV) -> None:
        self.stock_dir = stock_dir
        self.info_csv = info_csv

    def load_price_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for file_path in glob.glob(str(self.stock_dir / "*.csv")):
            ticker = Path(file_path).stem
            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue
            date_col = next(
                (candidate for candidate in ["Date", "Time", "Datetime", "timestamp"] if candidate in df.columns),
                None,
            )
            if date_col is None or "Close" not in df.columns:
                continue
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            if df[date_col].isna().all():
                continue
            df = df.dropna(subset=[date_col]).set_index(date_col)
            frames.append(df[["Close"]].rename(columns={"Close": ticker}))
        if not frames:
            return pd.DataFrame()
        prices = pd.concat(frames, axis=1)
        prices = prices.apply(pd.to_numeric, errors="coerce")
        prices = prices.sort_index()
        threshold = max(1, int(len(prices) * 0.5))
        prices = prices.dropna(axis=1, thresh=threshold)
        prices = prices.ffill().bfill().interpolate(method="time", limit_direction="both")
        prices = prices.dropna(axis=0, how="any")
        validate_price_matrix(prices)
        return prices

    def load_asset_metadata(self) -> dict[str, str]:
        if not self.info_csv.exists():
            return {}
        df_info = pd.read_csv(self.info_csv)
        symbol_col = "Symbol" if "Symbol" in df_info.columns else df_info.columns[0]
        name_col = "Asset" if "Asset" in df_info.columns else df_info.columns[min(1, len(df_info.columns) - 1)]
        extra_col = None
        if "Type" in df_info.columns:
            extra_col = "Type"
        elif len(df_info.columns) > 2:
            extra_col = df_info.columns[2]
        labels: dict[str, str] = {}
        for _, row in df_info.iterrows():
            ticker = str(row[symbol_col]).strip()
            name = str(row[name_col]).strip()
            extra = str(row[extra_col]).strip() if extra_col else ""
            labels[ticker] = f"{name} ({extra})" if extra else name
        return labels

    def latest_prices(self, prices: pd.DataFrame) -> pd.Series:
        return prices.iloc[-1].dropna()

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna(how="all")
        return returns

    def asset_list(self, prices: pd.DataFrame) -> list[str]:
        return list(prices.columns)
