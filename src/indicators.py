import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_realized_vol(df: pd.DataFrame, window: int = 20, annualization: int = 252) -> pd.DataFrame:
    """
    Rolling realized volatility from log returns (annualized).
    Used as a proxy for IV in Black-Scholes options pricing.
    """
    df = df.copy()
    rets = np.log(df["Close"]).diff()
    df["RV"] = rets.rolling(window).std() * (annualization ** 0.5)
    return df
