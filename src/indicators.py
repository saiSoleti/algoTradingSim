# src/indicators.py

import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(window=w).mean()
    return df
