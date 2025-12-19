from __future__ import annotations
import argparse
import sys

from src.data_loader import fetch_ohlcv
from src.indicators import add_moving_averages, add_realized_vol
from src.strategies import (
    sma_crossover_signals_from_df,
    sma_crossover_long_short_signals_from_df,
)
from src.backtester import (
    BacktestConfig,
    run_stock_backtest,
    run_long_call_backtest,
    run_stock_protective_put_backtest,
    buy_and_hold_equity,
)
from src.metrics import summarize
from src.plot import plot_equity


def parse_args():
    p = argparse.ArgumentParser(description="Algo Trading Simulator (Stocks + Options)")

    p.add_argument("--ticker", type=str, default="", help="Ticker or comma list (e.g. QQQ,SPY,AAPL)")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")

    p.add_argument("--mode", choices=["stock", "long_call", "protective_put"], default="stock")

    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)

    p.add_argument("--initial_cash", type=float, default=10_000.0)
    p.add_argument("--fee_bps", type=float, default=2.0)
    p.add_argument("--slip_bps", type=float, default=3.0)
    p.add_argument("--execution", choices=["next_open", "next_close"], default="next_open")

    p.add_argument("--dte", type=int, default=30)
    p.add_argument("--rv_window", type=int, default=20)

    return p.parse_args()


def clean_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def print_summary(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float) and (("CAGR" in k) or ("Drawdown" in k) or ("MaxDD" in k)):
            print(f"{k:>18}: {v: .2%}")
        elif isinstance(v, float):
            print(f"{k:>18}: {v:,.2f}")
        else:
            print(f"{k:>18}: {v}")


def prompt_instrument_mode() -> str:
    print("\nPick instrument mode:")
    print("1) stock")
    print("2) long_call (options)")
    print("3) protective_put (options)")
    choice = input("Enter choice [1-3]: ").strip()
    return {"1": "stock", "2": "long_call", "3": "protective_put"}.get(choice, "stock")


def prompt_position_type() -> str:
    print("\nPick position type (stock only):")
    print("1) long only")
    print("2) short only")
    print("3) long + short")
    choice = input("Enter choice [1-3]: ").strip()
    return {"1": "long", "2": "short", "3": "both"}.get(choice, "long")


def prompt_for_ticker() -> str:
    while True:
        raw = input("Enter ticker (or comma list like QQQ,SPY,AAPL): ").strip()
        if not raw:
            print("Please enter at least one ticker.")
            continue
        if raw.startswith("--"):
            print(
                "\nThat looks like command-line flags.\n"
                "Run flags like:\n"
                "  python main.py --ticker TSLA --mode long_call\n"
                "Or enter ONLY the ticker here (e.g. TSLA).\n"
            )
            continue
        return raw


def prompt_call_settings(cfg: BacktestConfig) -> BacktestConfig:
    print("\n=== Pick Your Call (Rules-Based Contract) ===")
    print("Choose DTE, ATM vs OTM %, number of contracts, and strike rounding.\n")

    dte = input(f"DTE in trading days (default {cfg.option_dte_days}): ").strip()
    if dte:
        cfg.option_dte_days = int(dte)

    otm = input("OTM % (0 for ATM, 0.05 for 5% OTM) [default 0]: ").strip()
    cfg.call_otm_pct = float(otm) if otm else 0.0

    contracts = input(f"# of contracts (default {cfg.call_contracts}): ").strip()
    cfg.call_contracts = int(contracts) if contracts else cfg.call_contracts

    step = input(f"Strike rounding step ($1 default, or $5, etc.) [default {cfg.option_strike_step}]: ").strip()
    if step:
        cfg.option_strike_step = float(step)

    print("\n[CALL SETTINGS CONFIRMED]")
    print("DTE:", cfg.option_dte_days)
    print("OTM %:", cfg.call_otm_pct)
    print("Contracts:", cfg.call_contracts)
    print("Strike step:", cfg.option_strike_step)

    return cfg


def main():
    args = parse_args()

    interactive = (args.ticker.strip() == "") and ("--ticker" not in sys.argv)

    # Interactive picks if user didn't pass --ticker
    pos_choice = "long"

    if interactive:
        args.mode = prompt_instrument_mode()
        if args.mode == "stock":
            pos_choice = prompt_position_type()
        else:
            pos_choice = "long"  # options are long-only in this version
        args.ticker = prompt_for_ticker()
    else:
        # CLI mode: keep args.mode as given; default pos_choice long unless user wants to extend CLI later
        pos_choice = "long"

    tickers = clean_tickers(args.ticker) or ["SPY"]

    cfg = BacktestConfig(
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        execution=args.execution,
        max_leverage=1.0,
        long_only=True,
        allow_short=False,
        option_dte_days=args.dte,
    )

    # Apply long/short selection (stock only)
    if args.mode == "stock":
        if pos_choice == "long":
            cfg.long_only = True
            cfg.allow_short = False
        elif pos_choice == "short":
            cfg.long_only = False
            cfg.allow_short = True
        else:  # both
            cfg.long_only = False
            cfg.allow_short = True
    else:
        # options modes long-only
        cfg.long_only = True
        cfg.allow_short = False

    # If long_call, ask user to pick call contract rules
    if args.mode == "long_call":
        cfg = prompt_call_settings(cfg)

    print("\n=== RUN CONFIG ===")
    print("tickers:", ", ".join(tickers))
    print("mode:", args.mode)
    if args.mode == "stock":
        print("position:", pos_choice)
    print("SMA:", args.fast, args.slow)
    print("execution:", args.execution)
    print("fee/slip bps:", args.fee_bps, args.slip_bps)
    print("RV window:", args.rv_window)

    for tkr in tickers:
        try:
            df = fetch_ohlcv(tkr, args.start, args.end)
        except Exception as e:
            print(f"\n[SKIP] {tkr}: {e}")
            continue

        df = add_moving_averages(df, windows=(args.fast, args.slow))
        df = add_realized_vol(df, window=args.rv_window)

        # choose signals
        if args.mode == "stock" and pos_choice in ("short", "both"):
            desired = sma_crossover_long_short_signals_from_df(df, fast=args.fast, slow=args.slow)
        else:
            desired = sma_crossover_signals_from_df(df, fast=args.fast, slow=args.slow)

        bench = buy_and_hold_equity(df, initial_cash=args.initial_cash)

        if args.mode == "stock":
            bt = run_stock_backtest(df, desired, cfg)
        elif args.mode == "long_call":
            bt = run_long_call_backtest(df, desired, cfg)
        else:
            bt = run_stock_protective_put_backtest(df, desired, cfg)

        stats = summarize(bt, benchmark_equity=bench)
        print_summary(f"{tkr} | {args.mode} | SMA({args.fast},{args.slow}) | {args.execution}", stats)
        plot_equity(bt, benchmark_equity=bench, title=f"{tkr} | {args.mode}")

if __name__ == "__main__":
    main()
