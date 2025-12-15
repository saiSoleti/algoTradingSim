# src/backtester.py

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 3.0
    execution: str = "next_open"   # "next_open" or "next_close"
    max_leverage: float = 1.0      # 1.0 = invest up to 100% of cash
    long_only: bool = True         # keep True for now

def _trade_cost(notional: float, fee_bps: float, slippage_bps: float) -> float:
    return notional * (fee_bps + slippage_bps) / 10_000.0

def run_backtest(ohlcv: pd.DataFrame, desired_pos: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Realistic backtest:
    - Signal on bar t-1 executes on bar t (avoids look-ahead bias)
    - Fees + slippage
    - Tracks cash, shares, equity
    - Long-only: desired position in {0,1}
    """
    df = ohlcv.copy()
    desired = desired_pos.reindex(df.index).fillna(0).astype(int)

    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("cfg.execution must be 'next_open' or 'next_close'")

    exec_price = df["Open"] if cfg.execution == "next_open" else df["Close"]

    cash = float(cfg.initial_cash)
    shares = 0.0

    cash_hist = [cash]
    shares_hist = [shares]
    equity_hist = [cash + shares * float(df["Close"].iloc[0])]
    trade_shares_hist = [0.0]
    trade_cost_hist = [0.0]

    for i in range(1, len(df)):
        # Execute on bar i, based on desired position from bar i-1
        target = int(desired.iloc[i - 1])
        if cfg.long_only and target < 0:
            target = 0

        px_exec = float(exec_price.iloc[i])
        px_close = float(df["Close"].iloc[i])

        current = 1 if shares > 0 else 0  # long-only exposure

        trade_shares = 0.0
        tcost = 0.0

        # Transition rules (long-only)
        if target == 1 and current == 0:
            # Buy with up to max_leverage of cash
            invest_cash = cash * cfg.max_leverage
            if px_exec <= 0:
                invest_cash = 0.0

            buy_shares = invest_cash / px_exec if px_exec > 0 else 0.0
            notional = buy_shares * px_exec
            tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
            total_outlay = notional + tcost

            if total_outlay > cash and px_exec > 0:
                # scale down to fit cash: x*px*(1+k) <= cash
                k = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
                buy_shares = cash / (px_exec * (1.0 + k))
                notional = buy_shares * px_exec
                tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
                total_outlay = notional + tcost

            cash -= total_outlay
            shares += buy_shares
            trade_shares = +buy_shares

        elif target == 0 and current == 1:
            # Sell all
            sell_shares = shares
            notional = sell_shares * px_exec
            tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)

            cash += notional - tcost
            shares = 0.0
            trade_shares = -sell_shares

        equity = cash + shares * px_close

        cash_hist.append(cash)
        shares_hist.append(shares)
        equity_hist.append(equity)
        trade_shares_hist.append(trade_shares)
        trade_cost_hist.append(tcost)

    out = pd.DataFrame(
        {
            "Close": df["Close"].values,
            "Cash": cash_hist,
            "Shares": shares_hist,
            "Equity": equity_hist,
            "TradeShares": trade_shares_hist,
            "TradeCost": trade_cost_hist,
        },
        index=df.index,
    )

    out["EquityReturn"] = out["Equity"].pct_change().fillna(0.0)

    return out

def buy_and_hold_equity(ohlcv: pd.DataFrame, initial_cash: float = 10_000.0) -> pd.Series:
    df = ohlcv.copy()
    first = float(df["Close"].iloc[0])
    if first <= 0:
        return pd.Series(index=df.index, data=np.nan, name="BuyHoldEquity")
    shares = initial_cash / first
    equity = shares * df["Close"]
    equity.name = "BuyHoldEquity"
    return equity
