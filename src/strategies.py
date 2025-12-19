from __future__ import annotations
import pandas as pd


def sma_crossover_signals_from_df(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Long-only desired position:
      1 when SMA(fast) > SMA(slow)
      0 otherwise
    """
    if fast >= slow:
        raise ValueError("fast must be < slow")

    fast_col = f"SMA{fast}"
    slow_col = f"SMA{slow}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(f"Missing columns: {fast_col}, {slow_col}. Run add_moving_averages first.")

    desired = (df[fast_col] > df[slow_col]).astype(int).fillna(0).astype(int)
    desired.name = "DesiredPos"
    return desired


def sma_crossover_long_short_signals_from_df(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Long + Short desired position:
      +1 when SMA(fast) > SMA(slow)
      -1 when SMA(fast) < SMA(slow)
       0 when equal / NaN zone
    """
    if fast >= slow:
        raise ValueError("fast must be < slow")

    fast_col = f"SMA{fast}"
    slow_col = f"SMA{slow}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(f"Missing columns: {fast_col}, {slow_col}. Run add_moving_averages first.")

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[df[fast_col] > df[slow_col]] = 1
    sig[df[fast_col] < df[slow_col]] = -1
    sig = sig.fillna(0).astype(int)
    sig.name = "DesiredPos"
    return sig
