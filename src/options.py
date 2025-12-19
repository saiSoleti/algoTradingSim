from __future__ import annotations
import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.0, option_type: str = "call") -> float:
    """
    Black-Scholes price for European call/put.

    S: underlying price
    K: strike
    T: time to expiry in years
    sigma: annualized volatility
    r: risk-free rate
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sigma = max(float(sigma), 1e-8)
    S = max(float(S), 1e-8)
    K = max(float(K), 1e-8)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def round_strike(S: float, step: float = 1.0) -> float:
    """Round strike to nearest step ($1 default)."""
    return round(S / step) * step
