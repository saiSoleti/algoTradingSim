from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


def _normalize_cached_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize cached CSV into a standard format:
    - Date index named 'Date'
    - Regular (non-MultiIndex) columns
    """
    # If Date column exists, use it as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        # Otherwise assume first column is the date index
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)
        df.index.name = "Date"

    # If columns somehow are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output into a standard format:
    - Flatten MultiIndex columns if present
    - Ensure Date index name is 'Date'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def download_price_history(
    ticker: str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
   
    os.makedirs(cache_dir, exist_ok=True)

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    csv_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path)
        return _normalize_cached_csv(cached)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,    
        group_by="column",     
        progress=False,
    )

    df = _normalize_yfinance_df(df)

    # Save to cache (with Date index)
    df.to_csv(csv_path)

    return df
