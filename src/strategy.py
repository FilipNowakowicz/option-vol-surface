"""ATM straddle P&L with full Greek attribution.

Setup
-----
At t0 we open a long ATM straddle on SPY: K = S(t0), maturity T0 days out.
We mark each leg daily using:
    * S(t)     = SPY close
    * sigma(t) = VIX(t)/100, used as a proxy for the ATM 30-day IV
                 (documented simplification: a true time series of full
                  surfaces is not available from yfinance)
    * T(t)     = T0 - (t - t0) / 365

For each step t-1 -> t we attribute the change in straddle value into:

    delta P&L  ~  delta(t-1)        * (S(t) - S(t-1))
    gamma P&L  ~  0.5 * gamma(t-1)  * (S(t) - S(t-1))**2
    vega  P&L  ~  vega(t-1)         * (sigma(t) - sigma(t-1))
    theta P&L  ~  theta(t-1)        * dt
    residual   =  total - sum(above)         (higher-order + cross-terms)

Both legs share strike and expiry, so we sum per-leg Greeks to get the
straddle Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import bs


@dataclass
class StraddleResult:
    daily: pd.DataFrame      # one row per re-mark day
    attribution: pd.DataFrame  # delta, gamma, vega, theta, residual, total per step
    totals: dict


def _straddle_greeks(S, K, T, r, sigma, q):
    """Sum of (call + put) for each Greek; gamma/vega/vanna/volga/veta
    are option-type-agnostic so are 2x the per-leg value."""
    return dict(
        price=bs.price(S, K, T, r, sigma, q, "call") + bs.price(S, K, T, r, sigma, q, "put"),
        delta=bs.delta(S, K, T, r, sigma, q, "call") + bs.delta(S, K, T, r, sigma, q, "put"),
        gamma=2.0 * bs.gamma(S, K, T, r, sigma, q),
        vega=2.0 * bs.vega(S, K, T, r, sigma, q),
        theta=bs.theta(S, K, T, r, sigma, q, "call") + bs.theta(S, K, T, r, sigma, q, "put"),
    )


def run_straddle(
    spy: pd.Series,
    vix: pd.Series,
    *,
    entry_date: pd.Timestamp,
    days_to_expiry: int = 60,
    holding_days: int = 21,
    r: float = 0.04,
    q: float = 0.0077,
) -> StraddleResult:
    """Simulate an ATM straddle held from entry_date for holding_days trading days.

    spy and vix must be pd.Series indexed by trading-day Timestamp.
    """
    spy = spy.dropna()
    vix = vix.dropna()
    spy, vix = spy.align(vix, join="inner")
    if entry_date not in spy.index:
        # snap to next available trading day
        candidates = spy.index[spy.index >= entry_date]
        if candidates.empty:
            raise ValueError(f"entry_date {entry_date} after available SPY data")
        entry_date = candidates[0]

    idx = spy.index
    entry_pos = idx.get_loc(entry_date)
    last_pos = min(entry_pos + holding_days, len(idx) - 1)

    S0 = float(spy.iloc[entry_pos])
    K = S0
    sigma0 = float(vix.iloc[entry_pos]) / 100.0
    T0 = days_to_expiry / 365.0

    daily_rows = []
    attr_rows = []

    # t0
    g0 = _straddle_greeks(S0, K, T0, r, sigma0, q)
    daily_rows.append({
        "date": entry_date, "S": S0, "sigma": sigma0, "T": T0,
        "value": g0["price"], "pnl": 0.0,
        "delta": g0["delta"], "gamma": g0["gamma"],
        "vega": g0["vega"], "theta": g0["theta"],
    })

    prev_S = S0
    prev_sigma = sigma0
    prev_T = T0
    prev_value = g0["price"]
    prev_greeks = g0

    for pos in range(entry_pos + 1, last_pos + 1):
        date = idx[pos]
        S = float(spy.iloc[pos])
        sigma = float(vix.iloc[pos]) / 100.0
        dt = (date - idx[pos - 1]).days / 365.0
        T = max(prev_T - dt, 1e-6)

        g = _straddle_greeks(S, K, T, r, sigma, q)
        total = g["price"] - prev_value

        dS = S - prev_S
        dsig = sigma - prev_sigma

        delta_pnl = prev_greeks["delta"] * dS
        gamma_pnl = 0.5 * prev_greeks["gamma"] * dS * dS
        vega_pnl = prev_greeks["vega"] * dsig
        theta_pnl = prev_greeks["theta"] * dt
        residual = total - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)

        daily_rows.append({
            "date": date, "S": S, "sigma": sigma, "T": T,
            "value": g["price"], "pnl": total,
            "delta": g["delta"], "gamma": g["gamma"],
            "vega": g["vega"], "theta": g["theta"],
        })
        attr_rows.append({
            "date": date, "total": total,
            "delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl, "theta_pnl": theta_pnl,
            "residual": residual,
        })

        prev_S, prev_sigma, prev_T, prev_value, prev_greeks = S, sigma, T, g["price"], g

    daily = pd.DataFrame(daily_rows).set_index("date")
    attribution = pd.DataFrame(attr_rows).set_index("date")

    total_pnl = daily["value"].iloc[-1] - daily["value"].iloc[0]
    sum_attr = attribution[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].sum().sum()
    total_residual = attribution["residual"].sum()
    # Gross attribution = sum of absolute Greek contributions. When total P&L
    # is near zero (Greeks cancelling), the residual-as-%-of-net explodes
    # spuriously, so we report residual-as-%-of-gross as well.
    gross_attr = attribution[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].abs().sum().sum()
    pct_residual_net = abs(total_residual) / max(abs(total_pnl), 1e-9)
    pct_residual_gross = abs(total_residual) / max(gross_attr, 1e-9)

    totals = dict(
        total_pnl=total_pnl,
        delta_pnl=attribution["delta_pnl"].sum(),
        gamma_pnl=attribution["gamma_pnl"].sum(),
        vega_pnl=attribution["vega_pnl"].sum(),
        theta_pnl=attribution["theta_pnl"].sum(),
        residual=total_residual,
        residual_pct_of_total=pct_residual_net,
        residual_pct_of_gross=pct_residual_gross,
        sum_attribution=sum_attr,
        gross_attribution=gross_attr,
        entry_S=daily["S"].iloc[0],
        exit_S=daily["S"].iloc[-1],
        entry_sigma=daily["sigma"].iloc[0],
        exit_sigma=daily["sigma"].iloc[-1],
        days_held=len(attribution),
        entry_premium=daily["value"].iloc[0],
        exit_value=daily["value"].iloc[-1],
    )

    return StraddleResult(daily=daily, attribution=attribution, totals=totals)
