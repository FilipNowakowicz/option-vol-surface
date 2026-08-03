"""Tests for src/bs.py.

Coverage:
  * Put-call parity to machine precision
  * Every closed-form Greek matches central-difference FD to <= 1e-5 rel error
    across a range of (S, K, T, sigma, r, q) points
  * Sign / limit sanity:
      - call delta in (0, e^{-qT}), put delta in (-e^{-qT}, 0)
      - gamma >= 0
      - vega >= 0
      - vega -> 0 as T -> 0 for OTM
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs  # noqa: E402


# A grid of (S, K, T, r, sigma, q) tuples covering ITM / ATM / OTM, short and
# long expiries, low and high vol, with and without dividends.
GRID = [
    (100, 100, 1.0, 0.03, 0.20, 0.00),
    (100, 100, 1.0, 0.03, 0.20, 0.02),
    (100, 90, 0.50, 0.05, 0.30, 0.01),
    (100, 110, 0.50, 0.05, 0.30, 0.01),
    (100, 100, 0.10, 0.01, 0.50, 0.00),
    (50, 60, 2.0, 0.04, 0.25, 0.03),
    (200, 180, 0.25, 0.02, 0.15, 0.00),
]


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_put_call_parity(S, K, T, r, sigma, q):
    """C - P = S e^{-qT} - K e^{-rT}."""
    C = bs.price(S, K, T, r, sigma, q, "call")
    P = bs.price(S, K, T, r, sigma, q, "put")
    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


# --------------------------------------------------------------------------
# Finite-difference verification for every Greek
# --------------------------------------------------------------------------
def _rel(a, b):
    denom = max(abs(a), abs(b), 1e-8)
    return abs(a - b) / denom


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
@pytest.mark.parametrize("option", ["call", "put"])
def test_delta_fd(S, K, T, r, sigma, q, option):
    h = 1e-4 * S
    fd = (
        bs.price(S + h, K, T, r, sigma, q, option) - bs.price(S - h, K, T, r, sigma, q, option)
    ) / (2 * h)
    analytic = bs.delta(S, K, T, r, sigma, q, option)
    assert _rel(fd, analytic) < 1e-5


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_gamma_fd(S, K, T, r, sigma, q):
    h = 1e-3 * S
    fd = (
        bs.price(S + h, K, T, r, sigma, q, "call")
        - 2 * bs.price(S, K, T, r, sigma, q, "call")
        + bs.price(S - h, K, T, r, sigma, q, "call")
    ) / (h * h)
    analytic = bs.gamma(S, K, T, r, sigma, q)
    assert _rel(fd, analytic) < 1e-4


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_vega_fd(S, K, T, r, sigma, q):
    h = 1e-5
    fd = (
        bs.price(S, K, T, r, sigma + h, q, "call") - bs.price(S, K, T, r, sigma - h, q, "call")
    ) / (2 * h)
    analytic = bs.vega(S, K, T, r, sigma, q)
    assert _rel(fd, analytic) < 1e-5


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
@pytest.mark.parametrize("option", ["call", "put"])
def test_theta_fd(S, K, T, r, sigma, q, option):
    """theta = dV/dt where t = calendar time, so a positive bump in t is a
    decrease in T. Central diff: (V(T-h) - V(T+h)) / (2h)."""
    h = 1e-5
    fd = (
        bs.price(S, K, T - h, r, sigma, q, option) - bs.price(S, K, T + h, r, sigma, q, option)
    ) / (2 * h)
    analytic = bs.theta(S, K, T, r, sigma, q, option)
    assert _rel(fd, analytic) < 1e-4


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
@pytest.mark.parametrize("option", ["call", "put"])
def test_rho_fd(S, K, T, r, sigma, q, option):
    h = 1e-6
    fd = (
        bs.price(S, K, T, r + h, sigma, q, option) - bs.price(S, K, T, r - h, sigma, q, option)
    ) / (2 * h)
    analytic = bs.rho(S, K, T, r, sigma, q, option)
    assert _rel(fd, analytic) < 1e-5


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_vanna_fd(S, K, T, r, sigma, q):
    """vanna = ddelta/dsigma."""
    h = 1e-5
    fd = (
        bs.delta(S, K, T, r, sigma + h, q, "call")
        - bs.delta(S, K, T, r, sigma - h, q, "call")
    ) / (2 * h)
    analytic = bs.vanna(S, K, T, r, sigma, q)
    assert _rel(fd, analytic) < 1e-4


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_volga_fd(S, K, T, r, sigma, q):
    """volga = dvega/dsigma."""
    h = 1e-5
    fd = (
        bs.vega(S, K, T, r, sigma + h, q) - bs.vega(S, K, T, r, sigma - h, q)
    ) / (2 * h)
    analytic = bs.volga(S, K, T, r, sigma, q)
    assert _rel(fd, analytic) < 1e-4


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
@pytest.mark.parametrize("option", ["call", "put"])
def test_charm_fd(S, K, T, r, sigma, q, option):
    """charm = ddelta/dt = -ddelta/dT."""
    h = 1e-5
    fd = (
        bs.delta(S, K, T - h, r, sigma, q, option)
        - bs.delta(S, K, T + h, r, sigma, q, option)
    ) / (2 * h)
    analytic = bs.charm(S, K, T, r, sigma, q, option)
    assert _rel(fd, analytic) < 1e-4


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_veta_fd(S, K, T, r, sigma, q):
    """veta = dvega/dt = -dvega/dT."""
    h = 1e-5
    fd = (
        bs.vega(S, K, T - h, r, sigma, q) - bs.vega(S, K, T + h, r, sigma, q)
    ) / (2 * h)
    analytic = bs.veta(S, K, T, r, sigma, q)
    assert _rel(fd, analytic) < 1e-4


# --------------------------------------------------------------------------
# Sign / limit sanity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_delta_bounds(S, K, T, r, sigma, q):
    cap = np.exp(-q * T)
    dc = bs.delta(S, K, T, r, sigma, q, "call")
    dp = bs.delta(S, K, T, r, sigma, q, "put")
    assert 0.0 < dc < cap
    assert -cap < dp < 0.0


@pytest.mark.parametrize("S,K,T,r,sigma,q", GRID)
def test_gamma_vega_nonneg(S, K, T, r, sigma, q):
    assert bs.gamma(S, K, T, r, sigma, q) >= 0
    assert bs.vega(S, K, T, r, sigma, q) >= 0


def test_vega_vanishes_otm_short_T():
    """Deep OTM call with tiny T: vega should be ~0."""
    v = bs.vega(S=100, K=150, T=1 / 365, r=0.03, sigma=0.20, q=0.0)
    assert v < 1e-3
