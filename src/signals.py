import pandas as pd

def sma_trend_signal(df: pd.DataFrame, ma_col="SMA50") -> pd.DataFrame:
    df = df.copy()
    df["signal"] = 0
    df.loc[df["Close"] > df[ma_col], "signal"] = 1
    return df
