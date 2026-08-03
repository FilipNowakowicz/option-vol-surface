"""Implied volatility from European option prices.

Primary solver: Newton-Raphson on f(sigma) = BS(sigma) - market_price = 0,
with f'(sigma) = vega. Fast (~3-5 iters in typical conditions) but unstable
when vega is small (deep OTM, very short T) or when the initial guess sits
outside the convergence basin.

Fallback: Brent's method on the same f over a bracket [sigma_lo, sigma_hi],
guaranteed to converge if a bracket exists.

No-arbitrage filter: prices outside the BS bounds return NaN, since no
sigma >= 0 reproduces them.

    Call bounds: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}
    Put  bounds: max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from . import bs

OptionType = Literal["call", "put"]

_TOL = 1e-9
_MAX_ITER = 60
_SIGMA_MIN = 1e-5
_SIGMA_MAX = 5.0


def _arbitrage_bounds(S, K, T, r, q, option: OptionType):
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option == "call":
        lo = max(S * disc_q - K * disc_r, 0.0)
        hi = S * disc_q
    else:
        lo = max(K * disc_r - S * disc_q, 0.0)
        hi = K * disc_r
    return lo, hi


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option: OptionType = "call",
) -> float:
    """Return implied vol or NaN if no valid root exists.

    Tries Newton-Raphson first, falls back to Brent.
    """
    if not (T > 0 and S > 0 and K > 0):
        return float("nan")

    lo, hi = _arbitrage_bounds(S, K, T, r, q, option)
    if market_price < lo - 1e-10 or market_price > hi + 1e-10:
        return float("nan")
    if market_price <= lo + 1e-12:
        return _SIGMA_MIN

    # --- Newton-Raphson ---
    # Brenner-Subrahmanyam initial guess; clamp to keep vega well-defined.
    sigma = math.sqrt(2 * math.pi / T) * market_price / S
    sigma = max(min(sigma, 2.0), 0.05)

    for _ in range(_MAX_ITER):
        p = bs.price(S, K, T, r, sigma, q, option)
        v = bs.vega(S, K, T, r, sigma, q)
        diff = p - market_price
        if abs(diff) < _TOL:
            return float(sigma)
        if v < 1e-10:  # vega vanishes -> abandon Newton
            break
        step = diff / v
        sigma_new = sigma - step
        if sigma_new <= _SIGMA_MIN or sigma_new >= _SIGMA_MAX:
            break  # stepped out of plausible range; Brent will handle
        if abs(sigma_new - sigma) < _TOL:
            return float(sigma_new)
        sigma = sigma_new

    # --- Brent fallback ---
    def f(s):
        return bs.price(S, K, T, r, s, q, option) - market_price

    f_lo = f(_SIGMA_MIN)
    f_hi = f(_SIGMA_MAX)
    if f_lo * f_hi > 0:
        return float("nan")  # no sign change -> no root in bracket
    try:
        root = brentq(f, _SIGMA_MIN, _SIGMA_MAX, xtol=_TOL, maxiter=200)
        return float(root)
    except Exception:
        return float("nan")


def implied_vol_vec(market_prices, S, K, T, r, q=0.0, option: OptionType = "call"):
    """Vectorised across (K, T, market_price, option) arrays of equal length.

    Used by surface.py to extract IV for an entire chain. q is scalar.
    """
    market_prices = np.asarray(market_prices, dtype=float)
    Ks = np.asarray(K, dtype=float)
    Ts = np.asarray(T, dtype=float)
    if isinstance(option, str):
        options = [option] * len(market_prices)
    else:
        options = list(option)
    out = np.full(len(market_prices), np.nan)
    for i, (mp, k, t, opt) in enumerate(zip(market_prices, Ks, Ts, options)):
        out[i] = implied_vol(mp, S, k, t, r, q, opt)
    return out
