# src/strategies.py

from __future__ import annotations
import pandas as pd


def sma_crossover_signals_from_df(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Desired position (0/1) based on SMA crossover.
    Assumes indicators.py already created SMA columns (SMA{fast}, SMA{slow}).
    """
    if fast >= slow:
        raise ValueError("fast must be < slow")

    fast_col = f"SMA{fast}"
    slow_col = f"SMA{slow}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(
            f"Missing columns: {fast_col}, {slow_col}. "
            "Run add_moving_averages(df, windows=(fast, slow)) first."
        )

    desired = (df[fast_col] > df[slow_col]).astype(int).fillna(0)
    desired.name = "DesiredPos"
    return desired


def momentum_signals_from_df(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Simple long/flat momentum.
    Long if Close[t] > Close[t-lookback], else flat.
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain 'Close'")

    mom = df["Close"] / df["Close"].shift(lookback) - 1.0
    desired = (mom > 0).astype(int).fillna(0)
    desired.name = "DesiredPos"
    return desired
