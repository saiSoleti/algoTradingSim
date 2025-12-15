# src/indicators.py

import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' to compute moving averages.")
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(window=w).mean()
    return df
