# src/plots.py

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(
    strategy_bt: pd.DataFrame,
    benchmark_equity: pd.Series | None = None,
    title: str = "Equity Curve",
) -> None:
    """
    Plots strategy equity vs optional buy-and-hold benchmark.

    Expects:
      strategy_bt index = dates
      strategy_bt["Equity"] = strategy equity series
      benchmark_equity = pd.Series indexed by dates (same index or reindexable)
    """
    if "Equity" not in strategy_bt.columns:
        raise ValueError("strategy_bt must contain an 'Equity' column")

    plt.figure()
    plt.plot(strategy_bt.index, strategy_bt["Equity"], label="Strategy")

    if benchmark_equity is not None:
        b = benchmark_equity.reindex(strategy_bt.index)
        plt.plot(strategy_bt.index, b, label="Buy & Hold")

    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_price_with_signals(
    df: pd.DataFrame,
    desired_pos: pd.Series,
    title: str = "Price + Desired Position",
) -> None:
    """
    Optional helper: plots Close price and overlays desired position (0/1) scaled.
    Useful for visually checking the crossover behavior.

    Expects:
      df["Close"] exists
      desired_pos is 0/1 indexed by df.index
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column")

    pos = desired_pos.reindex(df.index).fillna(0).astype(float)

    plt.figure()
    plt.plot(df.index, df["Close"], label="Close")

    # Scale position so it's visible on the same chart
    scaled = pos * float(df["Close"].max())
    plt.plot(df.index, scaled, label="DesiredPos (scaled)")

    plt.xlabel("Date")
    plt.ylabel("Price / Scaled Position")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
