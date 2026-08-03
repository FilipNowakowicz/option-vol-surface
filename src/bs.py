"""Black-Scholes-Merton analytical pricer with all first- and second-order Greeks.

Conventions
-----------
- T is time to expiry in years (T > 0).
- r is the continuously compounded risk-free rate.
- q is the continuous dividend yield.
- sigma is annualised volatility.
- "Time"-derivatives (theta, charm, veta) are with respect to *calendar time t*.
  Since T = T_expiry - t we have dT/dt = -1, so theta = -dV/dT, etc.
- Vega, vanna, volga are per unit change in sigma (not per 1% — caller
  scales if they want "per-1-vol-point").

All Greeks return a scalar (or numpy array, depending on input).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSInputs:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0


def _d1_d2(S, K, T, r, sigma, q):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def price(S, K, T, r, sigma, q=0.0, option: OptionType = "call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    if option == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def delta(S, K, T, r, sigma, q=0.0, option: OptionType = "call"):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if option == "call":
        return disc_q * norm.cdf(d1)
    if option == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError(option)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option: OptionType = "call"):
    """Per-calendar-year theta (dV/dt). Multiply by 1/365 for daily theta."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    base = -S * disc_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option == "call":
        return base - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    if option == "put":
        return base + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    raise ValueError(option)


def rho(S, K, T, r, sigma, q=0.0, option: OptionType = "call"):
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if option == "call":
        return K * T * disc_r * norm.cdf(d2)
    if option == "put":
        return -K * T * disc_r * norm.cdf(-d2)
    raise ValueError(option)


def vanna(S, K, T, r, sigma, q=0.0):
    """d2V / dS dsigma = dvega/dS = ddelta/dsigma. Same for call and put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return -disc_q * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, sigma, q=0.0):
    """d2V / dsigma2. Also called vomma. Same for call and put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma


def charm(S, K, T, r, sigma, q=0.0, option: OptionType = "call"):
    """ddelta/dt (calendar time). Sometimes called delta-decay."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    bracket = (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    common = -disc_q * norm.pdf(d1) * bracket
    if option == "call":
        return common + q * disc_q * norm.cdf(d1)
    if option == "put":
        return common - q * disc_q * norm.cdf(-d1)
    raise ValueError(option)


def veta(S, K, T, r, sigma, q=0.0):
    """dvega/dt (calendar time). Same for call and put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    inner = q + (r - q) * d1 / (sigma * np.sqrt(T)) - (1.0 + d1 * d2) / (2.0 * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T) * inner


def all_greeks(S, K, T, r, sigma, q=0.0, option: OptionType = "call") -> dict:
    """Return every Greek in one dict — convenient for the attribution exercise."""
    return {
        "price": price(S, K, T, r, sigma, q, option),
        "delta": delta(S, K, T, r, sigma, q, option),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, option),
        "rho": rho(S, K, T, r, sigma, q, option),
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
        "charm": charm(S, K, T, r, sigma, q, option),
        "veta": veta(S, K, T, r, sigma, q),
    }
