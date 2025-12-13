from src.data_loader import download_price_history
from src.backtester import run_backtest_all_in, max_drawdown
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ticker = input(
        "Enter a ticker (stock/ETF/crypto, e.g. PLTR, SPY, QQQ, BTC-USD): "
    ).strip().upper() or "PLTR"

    df = download_price_history(ticker, start="2024-01-01")

    # Extra safety: flatten MultiIndex if it slips through
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicators
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Signal: long if Close > SMA50 (trend filter)
    df["signal"] = 0
    df.loc[df["Close"] > df["SMA50"], "signal"] = 1

    # Backtest
    equity_curve, trades = run_backtest_all_in(df, initial_cash=10_000.0)

    # Metrics
    start_eq = float(equity_curve.iloc[0])
    end_eq = float(equity_curve.iloc[-1])
    total_return = end_eq / start_eq - 1.0
    mdd = max_drawdown(equity_curve)

    print(f"\nTicker: {ticker}")
    print(f"Initial equity: ${start_eq:,.2f}")
    print(f"Final equity:   ${end_eq:,.2f}")
    print(f"Total return:   {total_return*100:.2f}%")
    print(f"Max drawdown:   {mdd*100:.2f}%")
    print(f"Trades:         {len(trades)}")
    if trades:
        print("\nFirst 5 trades:")
        for t in trades[:5]:
            print(f"{t.date.date()} | {t.side} | shares={t.shares:.4f} @ ${t.price:.2f}")

    # Plot 1: Price + SMAs
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, df["SMA20"], label="SMA20", linestyle="--")
    plt.plot(df.index, df["SMA50"], label="SMA50", linestyle="--")
    plt.title(f"{ticker} Close + SMAs")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Equity curve
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(f"{ticker} Equity Curve (Signal: Close > SMA50)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
