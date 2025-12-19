from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(strategy_bt: pd.DataFrame, benchmark_equity: pd.Series | None = None, title: str = "Equity Curve") -> None:
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
