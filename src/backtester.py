from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    side: str   # "BUY" or "SELL"
    price: float
    shares: float


def run_backtest_all_in(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    price_col: str = "Close",
    signal_col: str = "signal",
) -> Tuple[pd.Series, List[Trade]]:
    """
    Simple long-only backtest:
    - signal=1 => be fully invested (all-in)
    - signal=0 => be in cash (all-out)
    - trades execute at the day's close
    """
    cash = float(initial_cash)
    shares = 0.0
    trades: List[Trade] = []

    equity_vals = []
    dates = []

    for date, row in df.iterrows():
        price = float(row[price_col])
        signal = int(row[signal_col])

        # Skip bad prices
        if price <= 0 or pd.isna(price):
            continue

        # BUY: want long but currently flat
        if signal == 1 and shares == 0:
            shares = cash / price
            cash = 0.0
            trades.append(Trade(date=date, side="BUY", price=price, shares=shares))

        # SELL: want cash but currently long
        elif signal == 0 and shares > 0:
            cash = shares * price
            trades.append(Trade(date=date, side="SELL", price=price, shares=shares))
            shares = 0.0

        equity = cash + shares * price
        dates.append(date)
        equity_vals.append(equity)

    equity_curve = pd.Series(equity_vals, index=pd.Index(dates, name="Date"), name="Equity")
    return equity_curve, trades


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown as a decimal (e.g., -0.25 means -25%).
    """
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())
