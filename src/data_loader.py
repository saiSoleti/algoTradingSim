from __future__ import annotations
import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol/date range.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected if c in df.columns]].dropna()

    if "Open" not in df.columns or "Close" not in df.columns:
        raise ValueError("Missing required columns: Open and Close are required.")

    df.index = pd.to_datetime(df.index)

    # Ensure Close is 1-D
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]

    return df
