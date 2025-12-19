from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.options import bs_price, round_strike


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 3.0
    execution: str = "next_open"  # "next_open" or "next_close"
    max_leverage: float = 1.0

    # stock position permissions
    long_only: bool = True
    allow_short: bool = False

    # options basics
    option_dte_days: int = 30
    option_contract_multiplier: int = 100
    option_strike_step: float = 1.0
    risk_free_rate: float = 0.0

    # user-picked call settings
    call_otm_pct: float = 0.0   # 0.0 = ATM, 0.05 = 5% OTM
    call_contracts: int = 1     # number of contracts


def _trade_cost(notional: float, fee_bps: float, slippage_bps: float) -> float:
    return notional * (fee_bps + slippage_bps) / 10_000.0


def buy_and_hold_equity(ohlcv: pd.DataFrame, initial_cash: float = 10_000.0) -> pd.Series:
    df = ohlcv.copy()
    first = float(df["Close"].iloc[0])
    shares = initial_cash / first if first > 0 else 0.0
    equity = shares * df["Close"]
    equity.name = "BuyHoldEquity"
    return equity


def run_stock_backtest(ohlcv: pd.DataFrame, desired_pos: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Supports desired_pos in {-1, 0, +1}:
      +1 => long
       0 => flat
      -1 => short (if cfg.allow_short True)

    Model:
      - Enter long: spend cash to buy shares
      - Enter short: sell borrowed shares -> cash increases
      - Equity = cash + shares * close (shares negative for short)
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
        target = int(desired.iloc[i - 1])

        # permission rules
        if cfg.long_only:
            target = 1 if target == 1 else 0
        if (target < 0) and (not cfg.allow_short):
            target = 0

        px_exec = float(exec_price.iloc[i])
        px_close = float(df["Close"].iloc[i])

        # current position from shares sign
        current = 0
        if shares > 0:
            current = 1
        elif shares < 0:
            current = -1

        trade_shares = 0.0
        tcost = 0.0

        # if target differs from current, we close current then open target
        if target != current and px_exec > 0:
            # 1) close current
            if current == 1:
                # sell all long shares
                notional = abs(shares) * px_exec
                tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
                cash += notional - tcost
                trade_shares -= shares
                shares = 0.0

            elif current == -1:
                # cover short: buy back shares
                notional = abs(shares) * px_exec
                tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
                cash -= notional + tcost
                trade_shares -= shares  # shares is negative; subtracting adds positive buy amount
                shares = 0.0

            # 2) open new target
            if target == 1:
                invest_cash = cash * cfg.max_leverage
                buy_shares = invest_cash / px_exec
                notional = buy_shares * px_exec
                open_cost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)

                total = notional + open_cost
                if total > cash:
                    k = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
                    buy_shares = cash / (px_exec * (1.0 + k))
                    notional = buy_shares * px_exec
                    open_cost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
                    total = notional + open_cost

                cash -= total
                shares += buy_shares
                trade_shares += buy_shares
                tcost += open_cost

            elif target == -1:
                # short shares sized off equity proxy (cash) for simplicity
                short_cash = cash * cfg.max_leverage
                short_shares = short_cash / px_exec
                notional = short_shares * px_exec
                open_cost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)

                # when shorting you RECEIVE notional, then pay costs
                cash += notional - open_cost
                shares -= short_shares  # negative shares
                trade_shares -= short_shares
                tcost += open_cost

        equity = cash + shares * px_close

        cash_hist.append(cash)
        shares_hist.append(shares)
        equity_hist.append(equity)
        trade_shares_hist.append(trade_shares)
        trade_cost_hist.append(tcost)

    out = pd.DataFrame(
        {
            "Close": df["Close"].to_numpy().reshape(-1),
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


def run_long_call_backtest(ohlcv: pd.DataFrame, desired_pos: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Long calls only:
      - Buy calls when signal turns ON
      - Sell when OFF or settle at expiry
    User chooses: DTE, OTM %, contracts, strike step.
    Requires df['RV'].
    """
    df = ohlcv.copy()
    desired = desired_pos.reindex(df.index).fillna(0).astype(int)

    if "RV" not in df.columns:
        raise ValueError("Options backtest requires df['RV'] (run add_realized_vol).")

    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("cfg.execution must be 'next_open' or 'next_close'")

    exec_price = df["Open"] if cfg.execution == "next_open" else df["Close"]

    cash = float(cfg.initial_cash)

    contracts = max(int(cfg.call_contracts), 1)
    has_call = False
    strike = 0.0
    expiry_i = -1

    cash_hist = [cash]
    opt_val_hist = [0.0]
    equity_hist = [cash]
    trade_cost_hist = [0.0]
    trade_count_hist = [0]

    for i in range(1, len(df)):
        px_exec = float(exec_price.iloc[i])
        px_close = float(df["Close"].iloc[i])

        sigma = float(df["RV"].iloc[i]) if not np.isnan(df["RV"].iloc[i]) else 0.25
        sigma = max(sigma, 0.05)

        tcost = 0.0
        trades = 0

        # settle at expiry at close
        if has_call and i >= expiry_i:
            intrinsic = max(px_close - strike, 0.0)
            cash += intrinsic * cfg.option_contract_multiplier * contracts
            has_call = False
            strike = 0.0
            expiry_i = -1

        # desired is 0/1 here (long calls only)
        target = int(desired.iloc[i - 1])
        target = 1 if target == 1 else 0

        # enter
        if target == 1 and not has_call:
            S_for_strike = px_exec * (1.0 + float(cfg.call_otm_pct))
            K = round_strike(S_for_strike, step=cfg.option_strike_step)

            T = cfg.option_dte_days / 252.0
            premium = bs_price(px_exec, K, T, sigma=sigma, r=cfg.risk_free_rate, option_type="call")

            notional = premium * cfg.option_contract_multiplier * contracts
            tcost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
            total = notional + tcost

            if total <= cash:
                cash -= total
                has_call = True
                strike = K
                expiry_i = min(i + cfg.option_dte_days, len(df) - 1)
                trades += 1

        # exit
        elif target == 0 and has_call:
            remaining_days = max(expiry_i - i, 0)
            T = remaining_days / 252.0
            mkt = bs_price(px_exec, strike, T, sigma=sigma, r=cfg.risk_free_rate, option_type="call")

            proceeds = mkt * cfg.option_contract_multiplier * contracts
            tcost = _trade_cost(proceeds, cfg.fee_bps, cfg.slippage_bps)

            cash += proceeds - tcost
            has_call = False
            strike = 0.0
            expiry_i = -1
            trades += 1

        # mark-to-market at close
        opt_val = 0.0
        if has_call:
            remaining_days = max(expiry_i - i, 0)
            T = remaining_days / 252.0
            mkt_close = bs_price(px_close, strike, T, sigma=sigma, r=cfg.risk_free_rate, option_type="call")
            opt_val = mkt_close * cfg.option_contract_multiplier * contracts

        equity = cash + opt_val

        cash_hist.append(cash)
        opt_val_hist.append(opt_val)
        equity_hist.append(equity)
        trade_cost_hist.append(tcost)
        trade_count_hist.append(trades)

    out = pd.DataFrame(
        {
            "Close": df["Close"].to_numpy().reshape(-1),
            "Cash": cash_hist,
            "OptionValue": opt_val_hist,
            "Equity": equity_hist,
            "TradeCount": trade_count_hist,
            "TradeCost": trade_cost_hist,
        },
        index=df.index,
    )
    out["EquityReturn"] = out["Equity"].pct_change().fillna(0.0)
    return out


def run_stock_protective_put_backtest(ohlcv: pd.DataFrame, desired_pos: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Long stock + buy 1 ATM put hedge (if cash left).
    Long-only by design.
    Requires df['RV'].
    """
    df = ohlcv.copy()
    desired = desired_pos.reindex(df.index).fillna(0).astype(int)

    if "RV" not in df.columns:
        raise ValueError("Protective put requires df['RV'] (run add_realized_vol).")

    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("cfg.execution must be 'next_open' or 'next_close'")

    exec_price = df["Open"] if cfg.execution == "next_open" else df["Close"]

    cash = float(cfg.initial_cash)
    shares = 0.0

    has_put = False
    put_strike = 0.0
    put_expiry_i = -1

    cash_hist = [cash]
    shares_hist = [shares]
    put_val_hist = [0.0]
    equity_hist = [cash]
    trade_cost_hist = [0.0]
    trade_count_hist = [0]

    for i in range(1, len(df)):
        px_exec = float(exec_price.iloc[i])
        px_close = float(df["Close"].iloc[i])

        sigma = float(df["RV"].iloc[i]) if not np.isnan(df["RV"].iloc[i]) else 0.25
        sigma = max(sigma, 0.05)

        tcost = 0.0
        trades = 0

        # settle put at expiry
        if has_put and i >= put_expiry_i:
            intrinsic = max(put_strike - px_close, 0.0)
            cash += intrinsic * cfg.option_contract_multiplier
            has_put = False
            put_strike = 0.0
            put_expiry_i = -1

        target = int(desired.iloc[i - 1])
        target = 1 if target == 1 else 0

        current = 1 if shares > 0 else 0

        if target == 1 and current == 0 and px_exec > 0:
            invest_cash = cash * cfg.max_leverage
            buy_shares = invest_cash / px_exec
            notional = buy_shares * px_exec
            stock_cost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
            total_outlay = notional + stock_cost

            if total_outlay > cash:
                k = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
                buy_shares = cash / (px_exec * (1.0 + k))
                notional = buy_shares * px_exec
                stock_cost = _trade_cost(notional, cfg.fee_bps, cfg.slippage_bps)
                total_outlay = notional + stock_cost

            cash -= total_outlay
            shares += buy_shares
            tcost += stock_cost
            trades += 1

            # buy ATM put if cash left
            K = round_strike(px_exec, step=cfg.option_strike_step)
            T = cfg.option_dte_days / 252.0
            premium = bs_price(px_exec, K, T, sigma=sigma, r=cfg.risk_free_rate, option_type="put")

            put_notional = premium * cfg.option_contract_multiplier
            put_cost = _trade_cost(put_notional, cfg.fee_bps, cfg.slippage_bps)

            if (put_notional + put_cost) <= cash:
                cash -= (put_notional + put_cost)
                has_put = True
                put_strike = K
                put_expiry_i = min(i + cfg.option_dte_days, len(df) - 1)
                tcost += put_cost
                trades += 1

        elif target == 0 and current == 1 and px_exec > 0:
            sell_notional = shares * px_exec
            stock_cost = _trade_cost(sell_notional, cfg.fee_bps, cfg.slippage_bps)
            cash += sell_notional - stock_cost
            shares = 0.0
            tcost += stock_cost
            trades += 1

            if has_put:
                remaining_days = max(put_expiry_i - i, 0)
                T = remaining_days / 252.0
                mkt = bs_price(px_exec, put_strike, T, sigma=sigma, r=cfg.risk_free_rate, option_type="put")

                proceeds = mkt * cfg.option_contract_multiplier
                put_cost = _trade_cost(proceeds, cfg.fee_bps, cfg.slippage_bps)

                cash += proceeds - put_cost
                has_put = False
                put_strike = 0.0
                put_expiry_i = -1
                tcost += put_cost
                trades += 1

        put_val = 0.0
        if has_put:
            remaining_days = max(put_expiry_i - i, 0)
            T = remaining_days / 252.0
            mkt_close = bs_price(px_close, put_strike, T, sigma=sigma, r=cfg.risk_free_rate, option_type="put")
            put_val = mkt_close * cfg.option_contract_multiplier

        equity = cash + shares * px_close + put_val

        cash_hist.append(cash)
        shares_hist.append(shares)
        put_val_hist.append(put_val)
        equity_hist.append(equity)
        trade_cost_hist.append(tcost)
        trade_count_hist.append(trades)

    out = pd.DataFrame(
        {
            "Close": df["Close"].to_numpy().reshape(-1),
            "Cash": cash_hist,
            "Shares": shares_hist,
            "PutValue": put_val_hist,
            "Equity": equity_hist,
            "TradeCount": trade_count_hist,
            "TradeCost": trade_cost_hist,
        },
        index=df.index,
    )
    out["EquityReturn"] = out["Equity"].pct_change().fillna(0.0)
    return out
