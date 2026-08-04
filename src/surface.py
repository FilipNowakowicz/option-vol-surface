"""Build an implied-vol surface from a cleaned option chain, characterise it.

For each (strike, expiry):
    * Convert mid -> implied vol via iv.py
    * Tag with log-moneyness k = ln(K / F), F = S e^{(r-q) T}
    * Drop quotes whose IV failed to extract (no-arbitrage violation)

Surface diagnostics:
    * Smile slices: IV vs k for each expiry
    * Term structure: ATM IV vs T (linear-interpolated to k=0)
    * Skew: slope of IV vs k near ATM per expiry
    * Butterfly: 25-delta call IV + 25-delta put IV - 2 ATM IV
    * Arbitrage sanity: call-price monotonicity in K within each expiry
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import iv


@dataclass
class SurfaceDiagnostics:
    smile: pd.DataFrame       # columns: T, expiry, k, iv, type, strike
    term_structure: pd.DataFrame  # T -> atm IV
    skew: pd.DataFrame        # T -> dIV/dk near ATM
    butterfly: pd.DataFrame   # T -> 25d C + 25d P - 2 ATM
    arb_violations: pd.DataFrame


def build_surface(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    """Attach IV and log-moneyness to a cleaned chain."""
    out = chain.copy()
    F = spot * np.exp((r - q) * out["T"])
    out["F"] = F
    out["k"] = np.log(out["strike"] / F)
    out["iv"] = iv.implied_vol_vec(
        market_prices=out["mid"].to_numpy(),
        S=spot,
        K=out["strike"].to_numpy(),
        T=out["T"].to_numpy(),
        r=r,
        q=q,
        option=out["type"].tolist(),
    )
    return out.dropna(subset=["iv"]).reset_index(drop=True)


def _atm_iv_per_expiry(surf: pd.DataFrame) -> pd.DataFrame:
    """Linear-interpolate IV to k=0 per expiry, separately for calls and puts.

    Use the *average* of the call-derived and put-derived ATM IV when both
    sides exist; this is more robust than picking either side.
    """
    rows = []
    for (T, exp), g in surf.groupby(["T", "expiry"], sort=True):
        atm_iv = _interp_at_k0(g)
        if not np.isnan(atm_iv):
            rows.append(dict(T=T, expiry=exp, atm_iv=atm_iv))
    return pd.DataFrame(rows, columns=["T", "expiry", "atm_iv"]).sort_values("T").reset_index(drop=True)


def _interp_at_k0(g: pd.DataFrame) -> float:
    """Linear-interpolate IV at k=0 from sorted (k, iv) points.

    Uses an OTM-preferred composite: calls for k > 0, puts for k < 0 (cleanest
    market quotes), then linearly interpolates between the two closest
    bracketing strikes.
    """
    use = pd.concat(
        [g[(g["type"] == "call") & (g["k"] >= 0)], g[(g["type"] == "put") & (g["k"] < 0)]]
    )
    if len(use) < 2:
        use = g
    if len(use) < 2:
        return float("nan")
    use = use.sort_values("k")
    ks = use["k"].to_numpy()
    ivs = use["iv"].to_numpy()
    if ks.min() > 0 or ks.max() < 0:
        return float("nan")  # no bracket
    return float(np.interp(0.0, ks, ivs))


def _skew_per_expiry(surf: pd.DataFrame, band: float = 0.10) -> pd.DataFrame:
    """OLS slope of IV on k restricted to |k| <= band, per expiry."""
    rows = []
    for (T, exp), g in surf.groupby(["T", "expiry"], sort=True):
        sub = g[np.abs(g["k"]) <= band]
        # Combine call/put on the same side: prefer OTM (calls for k>0, puts for k<0)
        otm = pd.concat(
            [sub[(sub["type"] == "call") & (sub["k"] >= 0)],
             sub[(sub["type"] == "put") & (sub["k"] < 0)]]
        )
        if len(otm) < 3:
            continue
        k = otm["k"].to_numpy()
        v = otm["iv"].to_numpy()
        slope = np.polyfit(k, v, 1)[0]
        rows.append(dict(T=T, expiry=exp, skew_dIVdk=slope))
    return pd.DataFrame(rows, columns=["T", "expiry", "skew_dIVdk"]).sort_values("T").reset_index(drop=True)


def _butterfly_per_expiry(
    surf: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    target_delta: float = 0.25,
) -> pd.DataFrame:
    """For each expiry, pick the call closest to +target_delta and the put
    closest to -target_delta; report 25dC_IV + 25dP_IV - 2 ATM_IV."""
    from . import bs

    rows = []
    atm = _atm_iv_per_expiry(surf).set_index("T")["atm_iv"]
    for (T, exp), g in surf.groupby(["T", "expiry"], sort=True):
        calls = g[g["type"] == "call"].copy()
        puts = g[g["type"] == "put"].copy()
        if calls.empty or puts.empty:
            continue
        calls["delta"] = [
            bs.delta(spot, k, T, r, s, q, "call")
            for k, s in zip(calls["strike"], calls["iv"])
        ]
        puts["delta"] = [
            bs.delta(spot, k, T, r, s, q, "put")
            for k, s in zip(puts["strike"], puts["iv"])
        ]
        c_row = calls.iloc[(calls["delta"] - target_delta).abs().argsort().iloc[0]]
        p_row = puts.iloc[(puts["delta"] + target_delta).abs().argsort().iloc[0]]
        if T not in atm.index:
            continue
        rows.append(
            dict(
                T=T, expiry=exp,
                call25_iv=c_row["iv"],
                put25_iv=p_row["iv"],
                atm_iv=atm.loc[T],
                butterfly=c_row["iv"] + p_row["iv"] - 2 * atm.loc[T],
            )
        )
    return pd.DataFrame(
        rows,
        columns=["T", "expiry", "call25_iv", "put25_iv", "atm_iv", "butterfly"],
    ).sort_values("T").reset_index(drop=True)


def _arbitrage_violations(surf: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    """Report simple per-quote bounds and vertical-spread violations.

    These checks are deliberately diagnostic rather than a repair mechanism:
    quotes are never smoothed or discarded here.  They cover the European
    price bounds and monotonicity in strike for calls and puts separately.
    """
    rows = []
    for (T, exp, option), group in surf.groupby(["T", "expiry", "type"], sort=True):
        g = group.sort_values("strike")
        discounted_spot = spot * np.exp(-q * T)
        discounted_strike = g["strike"] * np.exp(-r * T)
        if option == "call":
            lower = np.maximum(discounted_spot - discounted_strike, 0.0)
            upper = np.full(len(g), discounted_spot)
        else:
            lower = np.maximum(discounted_strike - discounted_spot, 0.0)
            upper = discounted_strike.to_numpy()
        for row, lo, hi in zip(g.itertuples(index=False), lower, upper):
            if row.mid < lo - 1e-6:
                rows.append(dict(T=T, expiry=exp, type=option, check="lower_bound", strike=row.strike, mid=row.mid))
            if row.mid > hi + 1e-6:
                rows.append(dict(T=T, expiry=exp, type=option, check="upper_bound", strike=row.strike, mid=row.mid))
        diffs = g["mid"].diff()
        # Calls decrease and puts increase as strike rises.
        bad = g[diffs > 1e-6] if option == "call" else g[diffs < -1e-6]
        for row in bad.itertuples(index=False):
            rows.append(dict(T=T, expiry=exp, type=option, check="vertical_monotonicity", strike=row.strike, mid=row.mid))
    return pd.DataFrame(rows, columns=["T", "expiry", "type", "check", "strike", "mid"])


def diagnose(
    surf: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
) -> SurfaceDiagnostics:
    return SurfaceDiagnostics(
        smile=surf[["T", "expiry", "k", "iv", "type", "strike"]].copy(),
        term_structure=_atm_iv_per_expiry(surf),
        skew=_skew_per_expiry(surf),
        butterfly=_butterfly_per_expiry(surf, spot, r, q),
        arb_violations=_arbitrage_violations(surf, spot, r, q),
    )
