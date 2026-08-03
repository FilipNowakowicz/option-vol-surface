"""Implied-volatility round-trip and edge-case tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs, iv  # noqa: E402


# (S, K, T, r, sigma_true, q, option)
ROUND_TRIP_GRID = [
    (100, 100, 1.0, 0.03, 0.20, 0.00, "call"),
    (100, 100, 1.0, 0.03, 0.20, 0.00, "put"),
    (100, 90, 0.5, 0.05, 0.30, 0.02, "call"),
    (100, 110, 0.5, 0.05, 0.30, 0.02, "put"),
    (100, 80, 0.25, 0.01, 0.40, 0.00, "call"),
    (100, 120, 0.25, 0.01, 0.40, 0.00, "put"),
    (100, 100, 2.0, 0.04, 0.15, 0.01, "call"),
    (50, 60, 1.5, 0.03, 0.55, 0.00, "put"),
]


@pytest.mark.parametrize("S,K,T,r,sigma_true,q,option", ROUND_TRIP_GRID)
def test_iv_round_trip(S, K, T, r, sigma_true, q, option):
    """Price -> implied vol must recover the input sigma to ~1e-6."""
    p = bs.price(S, K, T, r, sigma_true, q, option)
    sigma_hat = iv.implied_vol(p, S, K, T, r, q, option)
    assert abs(sigma_hat - sigma_true) < 1e-6


def test_iv_rejects_below_lower_bound():
    """A call priced below intrinsic must return NaN."""
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.0
    lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    out = iv.implied_vol(lower - 0.5, S, K, T, r, q, "call")
    assert math.isnan(out)


def test_iv_rejects_above_upper_bound():
    """A call priced above S e^{-qT} must return NaN."""
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.0
    out = iv.implied_vol(S * math.exp(-q * T) + 1.0, S, K, T, r, q, "call")
    assert math.isnan(out)


def test_iv_deep_otm_short_T():
    """Deep OTM with short T: vega is small, exercising the Brent fallback.
    Still must recover sigma to a reasonable tolerance."""
    S, K, T, r, q = 100.0, 115.0, 14 / 365, 0.03, 0.0
    sigma_true = 0.30
    p = bs.price(S, K, T, r, sigma_true, q, "call")
    assert p > 1e-6, "test point too extreme — price below numerical noise"
    sigma_hat = iv.implied_vol(p, S, K, T, r, q, "call")
    assert not math.isnan(sigma_hat)
    assert abs(sigma_hat - sigma_true) < 1e-4


def test_iv_vec():
    """Vectorised version returns one IV per strike."""
    S, T, r, q = 100.0, 0.5, 0.03, 0.0
    sigma_true = 0.25
    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = np.array([bs.price(S, K, T, r, sigma_true, q, "call") for K in Ks])
    out = iv.implied_vol_vec(prices, S, Ks, np.full_like(Ks, T), r, q, "call")
    assert np.allclose(out, sigma_true, atol=1e-6)
