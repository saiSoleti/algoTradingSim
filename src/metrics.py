# src/metrics.py

from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)

def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    vol = r.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / vol)

def summarize(bt: pd.DataFrame, benchmark_equity: pd.Series | None = None) -> dict:
    eq = bt["Equity"]
    rets = bt["EquityReturn"]

    out = {
        "Final Equity": float(eq.iloc[-1]),
        "CAGR": cagr(eq),
        "Max Drawdown": max_drawdown(eq),
        "Sharpe": sharpe(rets),
        "Num Trades": int((bt["TradeShares"] != 0).sum()),
        "Total Costs": float(bt["TradeCost"].sum()),
    }

    if benchmark_equity is not None:
        b = benchmark_equity.reindex(bt.index).dropna()
        if len(b) > 1:
            out["Benchmark Final"] = float(b.iloc[-1])
            out["Benchmark CAGR"] = cagr(b)
            out["Benchmark MaxDD"] = max_drawdown(b)

    return out
