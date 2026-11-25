"""Monte Carlo pricer under GBM with variance reduction.

Two layered techniques:
    1. Antithetic variates  — pair Z with -Z; halves variance "for free".
    2. Control variate      — use the discounted terminal underlying e^{-rT} S_T,
                               whose expectation is the analytical S_0 e^{-qT},
                               with optimal coefficient estimated in-sample.

Returns mean price and standard error so the convergence study has error bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int


def _terminal_prices(S, T, r, q, sigma, n_pairs, rng):
    """Return S_T for 2*n_pairs paths using antithetic Z."""
    Z = rng.standard_normal(n_pairs)
    drift = (r - q - 0.5 * sigma * sigma) * T
    diff = sigma * np.sqrt(T)
    up = S * np.exp(drift + diff * Z)
    dn = S * np.exp(drift - diff * Z)
    return np.concatenate([up, dn])


def _pair_means(values, n_pairs):
    """Average each path with its antithetic partner (first half vs second half)."""
    return 0.5 * (values[:n_pairs] + values[n_pairs:])


def _payoff(ST, K, option: OptionType):
    if option == "call":
        return np.maximum(ST - K, 0.0)
    if option == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError(option)


def price_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: OptionType = "call",
    n_paths: int = 200_000,
    seed: int | None = None,
    control_variate: bool = True,
) -> MCResult:
    """Price a European option by GBM Monte Carlo.

    n_paths must be even (antithetic pairing).
    """
    if n_paths % 2:
        n_paths += 1
    n_pairs = n_paths // 2
    rng = np.random.default_rng(seed)

    ST = _terminal_prices(S, T, r, q, sigma, n_pairs, rng)
    disc = np.exp(-r * T)
    # Antithetic paths are dependent in pairs, so the independent samples are
    # the pair means. Estimating the control coefficient or the standard
    # error from individual paths would treat 2n correlated draws as 2n
    # independent ones and misstate both.
    X = _pair_means(disc * _payoff(ST, K, option), n_pairs)

    if control_variate:
        # Y = disc * ST; E[Y] = S exp(-qT)
        Y = _pair_means(disc * ST, n_pairs)
        EY = S * np.exp(-q * T)
        # Optimal c on the pair means: c_hat = Cov(X, Y) / Var(Y)
        cov = np.cov(X, Y, ddof=1)
        c_hat = cov[0, 1] / cov[1, 1]
        X_adj = X - c_hat * (Y - EY)
    else:
        X_adj = X

    mean = float(np.mean(X_adj))
    se = float(np.std(X_adj, ddof=1) / np.sqrt(n_pairs))
    return MCResult(price=mean, stderr=se, n_paths=2 * n_pairs)


def price_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: OptionType = "call",
    n_paths: int = 100_000,
    n_steps: int = 50,
    seed: int | None = None,
) -> MCResult:
    """Arithmetic-average Asian option, MC only (no closed form).

    Stretch goal: shown to demonstrate path-dependent pricing, but no
    benchmark — the standard error is the entire credibility statement.
    """
    if n_paths % 2:
        n_paths += 1
    n_pairs = n_paths // 2
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diff = sigma * np.sqrt(dt)
    Z = rng.standard_normal((n_pairs, n_steps))

    # Antithetic block: build both Z and -Z paths simultaneously.
    increments_pos = drift + diff * Z
    increments_neg = drift - diff * Z
    log_paths_pos = np.cumsum(increments_pos, axis=1)
    log_paths_neg = np.cumsum(increments_neg, axis=1)
    paths_pos = S * np.exp(log_paths_pos)
    paths_neg = S * np.exp(log_paths_neg)
    paths = np.concatenate([paths_pos, paths_neg], axis=0)

    avg = paths.mean(axis=1)
    disc = np.exp(-r * T)
    X = _pair_means(disc * _payoff(avg, K, option), n_pairs)
    mean = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n_pairs))
    return MCResult(price=mean, stderr=se, n_paths=2 * n_pairs)
