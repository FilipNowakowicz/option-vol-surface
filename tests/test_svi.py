"""SVI fitting is deterministic, constrained, and explicit about diagnostics."""

import numpy as np
import pandas as pd

from src import bs
from src.svi import calibrate_slice, deterministic_split, surface_arbitrage_diagnostics, total_variance


def _synthetic_slice() -> pd.DataFrame:
    T, spot, r, q = 0.5, 100.0, 0.02, 0.01
    k = np.linspace(-0.30, 0.30, 21)
    a, b, rho, m, sigma = 0.012, 0.18, -0.45, 0.02, 0.16
    w = total_variance(k, a, b, rho, m, sigma)
    iv = np.sqrt(w / T)
    forward = spot * np.exp((r - q) * T)
    strike = forward * np.exp(k)
    kind = np.where(k < 0, "put", "call")
    mid = np.array([bs.price(spot, K, T, r, vol, q, option) for K, vol, option in zip(strike, iv, kind)])
    return pd.DataFrame({
        "T": T, "expiry": pd.Timestamp("2027-01-15"), "k": k, "iv": iv,
        "strike": strike, "type": kind, "mid": mid,
    })


def test_deterministic_split_keeps_endpoints_and_is_repeatable():
    quotes = _synthetic_slice()
    train, holdout = deterministic_split(quotes)
    train_again, holdout_again = deterministic_split(quotes)
    assert train["strike"].min() == quotes["strike"].min()
    assert train["strike"].max() == quotes["strike"].max()
    assert holdout["strike"].tolist() == holdout_again["strike"].tolist()
    assert train["strike"].tolist() == train_again["strike"].tolist()


def test_svi_recovers_clean_smile_and_enforces_positive_total_variance():
    result = calibrate_slice(_synthetic_slice(), spot=100.0, r=0.02, q=0.01)
    assert result is not None
    assert result.success
    assert result.holdout_iv_rmse < 1e-4
    assert result.holdout_price_rmse < 1e-3
    grid = np.linspace(-3.0, 3.0, 301)
    assert np.all(total_variance(grid, result.a, result.b, result.rho, result.m, result.sigma) > 0)


def test_surface_diagnostics_label_grid_checks_not_constraints():
    fits = pd.DataFrame([
        {"T": 0.25, "a": 0.01, "b": 0.1, "rho": -0.4, "m": 0.0, "sigma": 0.2, "convexity_violations": 2},
        {"T": 0.50, "a": 0.001, "b": 0.1, "rho": -0.4, "m": 0.0, "sigma": 0.2, "convexity_violations": 0},
    ])
    diagnostics = surface_arbitrage_diagnostics(fits)
    assert diagnostics["butterfly_grid_violations"] == 2
    assert diagnostics["calendar_grid_violations"] > 0
