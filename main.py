import argparse

from src.data_loader import fetch_ohlcv
from src.indicators import add_moving_averages
from src.strategies import sma_crossover_signals_from_df, momentum_signals_from_df
from src.backtester import run_backtest, BacktestConfig, buy_and_hold_equity
from src.metrics import summarize
from src.plot import plot_equity
from src.strategies import sma_crossover_signals_from_df, momentum_signals_from_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--strategy", choices=["sma", "momentum"], default="sma")

    # strategy params
    p.add_argument("--fast", type=int, default=20)     # sma
    p.add_argument("--slow", type=int, default=50)     # sma
    p.add_argument("--lookback", type=int, default=20) # momentum

    # execution/cost params
    p.add_argument("--initial_cash", type=float, default=10_000.0)
    p.add_argument("--fee_bps", type=float, default=2.0)
    p.add_argument("--slip_bps", type=float, default=3.0)
    p.add_argument("--execution", choices=["next_open", "next_close"], default="next_open")
    return p.parse_args()

def main():
    args = parse_args()

    df = fetch_ohlcv(args.ticker, args.start, args.end)

    # indicators needed depend on strategy
    if args.strategy == "sma":
        df = add_moving_averages(df, windows=(args.fast, args.slow))
        desired = sma_crossover_signals_from_df(df, fast=args.fast, slow=args.slow)
    elif args.strategy == "momentum":
        desired = momentum_signals_from_df(df, lookback=args.lookback)
    else:
        raise ValueError("Unknown strategy")

    cfg = BacktestConfig(
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        execution=args.execution,
        max_leverage=1.0,
        long_only=True,
    )

    bt = run_backtest(df, desired, cfg)
    bench = buy_and_hold_equity(df, initial_cash=args.initial_cash)
    stats = summarize(bt, benchmark_equity=bench)

    print("\n=== Backtest Summary ===")
    for k, v in stats.items():
        if isinstance(v, float) and ("CAGR" in k or "Drawdown" in k or "MaxDD" in k):
            print(f"{k:>18}: {v: .2%}")
        elif isinstance(v, float):
            print(f"{k:>18}: {v:,.2f}")
        else:
            print(f"{k:>18}: {v}")

    plot_equity(bt, benchmark_equity=bench, title=f"{args.ticker} | {args.strategy} | {args.execution}")

if __name__ == "__main__":
    main()
