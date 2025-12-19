Algorithmic Trading Simulator (Stocks & Options)

A modular Python-based trading simulator built to explore signal-based strategies, execution mechanics, and risk expression across instruments (stocks vs options).

The project emphasizes understanding market behavior over curve-fitting.

Features:
    Historical OHLCV data via yfinance
    SMA crossover strategies (configurable fast/slow)
    Long-only, short-only, and long+short stock trading
    Event-driven backtesting with fees and slippage
    Options strategies (educational):
    Long Calls (convex upside)
    Protective Puts (downside insurance)
    Black–Scholes pricing using realized volatility proxy
    Performance metrics: CAGR, max drawdown, Sharpe
    Equity curve visualization vs buy-and-hold
    Architecture
    Data → Indicators → Signals → Trades → Equity → Metrics → Plots


Key idea:

Signals create intent.
Execution adds realism.
Instruments reshape risk.

Project Structure
algoTradingSim/
├── main.py
└── src/
    ├── data_loader.py
    ├── indicators.py
    ├── strategies.py
    ├── backtester.py
    ├── options.py
    ├── metrics.py
    └── plot.py

How to Run
Interactive (recommended)
python main.py


You’ll choose:

Instrument (stock / options)

Position type (long / short / both)

Ticker(s)

(Options) contract parameters

CLI
python main.py --ticker QQQ --mode stock
python main.py --ticker TSLA --mode long_call

Market Insights:
    Trend-following works best in strong markets (e.g. QQQ)
    Choppy markets increase trade count and drawdowns
    Options do not improve signals — they change payoff shape
    Protective puts reduce drawdowns at the cost of returns

Limitations:
    Uses realized volatility as IV proxy
    No bid–ask spreads or liquidity modeling
    European options only
    No margin or assignment risk