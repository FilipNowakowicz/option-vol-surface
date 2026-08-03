"""SPY option chain fetch + caching.

yfinance gives us the full chain at all listed expiries for the current
snapshot. We cache the resulting frame as Parquet so analysis is reproducible
even after the live chain changes.

For a research-project snapshot, a single trade-date capture is enough.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_MAX_AGE_S = 24 * 3600  # one day


def _spot_and_yield(ticker: str) -> tuple[float, float]:
    """Spot price and continuous dividend yield from yfinance."""
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", auto_adjust=False)
    spot = float(hist["Close"].iloc[-1])
    info = t.info or {}
    # yfinance reports either trailingAnnualDividendYield (decimal) or
    # dividendYield (sometimes %, sometimes decimal). Be defensive.
    q = info.get("trailingAnnualDividendYield") or info.get("dividendYield") or 0.0
    if q > 1.0:  # some yfinance versions return percent
        q = q / 100.0
    return spot, float(q)


def fetch_chain(ticker: str = "SPY", force: bool = False) -> dict:
    """Return {spot, q, asof, chain DataFrame} for the current snapshot.

    DataFrame columns:
        type            'call' | 'put'
        strike          float
        expiry          pd.Timestamp (date, UTC midnight)
        T               years to expiry (calendar days / 365)
        bid, ask, mid   floats
        volume, open_interest
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_chain.parquet"
    meta_path = CACHE_DIR / f"{ticker}_chain_meta.parquet"

    fresh = (
        cache_path.exists()
        and (time.time() - cache_path.stat().st_mtime) < CACHE_MAX_AGE_S
    )
    if fresh and not force:
        df = pd.read_parquet(cache_path)
        meta = pd.read_parquet(meta_path).iloc[0]
        return dict(
            spot=float(meta["spot"]),
            q=float(meta["q"]),
            asof=pd.Timestamp(meta["asof"]),
            chain=df,
        )

    spot, q = _spot_and_yield(ticker)
    asof = pd.Timestamp(datetime.utcnow().date())
    t = yf.Ticker(ticker)
    expiries = list(t.options)

    frames = []
    for exp_str in expiries:
        try:
            chain = t.option_chain(exp_str)
        except Exception:
            continue
        exp = pd.Timestamp(exp_str)
        T = max((exp - asof).days, 1) / 365.0
        for kind, df in [("call", chain.calls), ("put", chain.puts)]:
            d = df.copy()
            d["type"] = kind
            d["expiry"] = exp
            d["T"] = T
            frames.append(d)

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "type": raw["type"],
            "strike": raw["strike"].astype(float),
            "expiry": raw["expiry"],
            "T": raw["T"].astype(float),
            "bid": raw["bid"].astype(float),
            "ask": raw["ask"].astype(float),
            "mid": ((raw["bid"] + raw["ask"]) / 2.0).astype(float),
            "volume": raw["volume"].fillna(0).astype(float),
            "open_interest": raw.get("openInterest", pd.Series(np.zeros(len(raw))))
            .fillna(0)
            .astype(float),
        }
    )

    out.to_parquet(cache_path)
    pd.DataFrame([{"spot": spot, "q": q, "asof": asof}]).to_parquet(meta_path)

    return dict(spot=spot, q=q, asof=asof, chain=out)


def fetch_history(ticker: str, start: str = "2025-01-01") -> pd.Series:
    """Daily close series, cached as Parquet."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    path = CACHE_DIR / f"{safe}_hist.parquet"
    if path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_S:
        return pd.read_parquet(path)[safe]

    data = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = safe
    close.to_frame(name=safe).to_parquet(path)
    return close


def clean_chain(
    df: pd.DataFrame,
    *,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.25,
    min_oi: float = 10,
) -> pd.DataFrame:
    """Drop quotes unsafe for IV extraction.

    Filters:
      * bid > min_bid              — eliminates penny/zero-bid quotes
      * (ask - bid)/mid <= max_rel_spread  — eliminates illiquid wide markets
      * open interest >= min_oi    — eliminates dead strikes
      * 5d <= T <= 2y              — too-near and too-far expiries are noisy
    """
    d = df.copy()
    d = d[d["bid"] > min_bid]
    d = d[d["ask"] > d["bid"]]
    rel_spread = (d["ask"] - d["bid"]) / d["mid"].clip(lower=1e-6)
    d = d[rel_spread <= max_rel_spread]
    d = d[d["open_interest"] >= min_oi]
    d = d[(d["T"] >= 5 / 365) & (d["T"] <= 2.0)]
    return d.reset_index(drop=True)
