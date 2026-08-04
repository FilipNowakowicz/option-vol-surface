"""SPY option-chain data and explicitly refreshed runtime caches.

The committed option-chain files are a frozen research artefact.  Normal
runs always read them, so the figures and numbers in the repository cannot be
silently changed by a live yfinance response.  A live chain is available only
through ``refresh=True`` and is written under ``cache/live/`` rather than over
the committed snapshot.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


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


def _read_chain(cache_path: Path, meta_path: Path) -> dict:
    """Read one option-chain snapshot and its metadata."""
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"missing option-chain snapshot: {cache_path} and {meta_path} are required"
        )
    meta = pd.read_parquet(meta_path).iloc[0]
    return dict(
        spot=float(meta["spot"]),
        q=float(meta["q"]),
        asof=pd.Timestamp(meta["asof"]),
        chain=pd.read_parquet(cache_path),
    )


def fetch_chain(ticker: str = "SPY", *, refresh: bool = False) -> dict:
    """Return the frozen chain snapshot, or explicitly fetch a new live one.

    ``refresh=False`` (the default) reads the committed snapshot at
    ``cache/<ticker>_chain.parquet``.  ``refresh=True`` calls yfinance and
    writes a date-stamped, ignored capture below ``cache/live/``.  It never
    overwrites the committed research evidence.

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
    if not refresh:
        return _read_chain(cache_path, meta_path)

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

    live_dir = CACHE_DIR / "live"
    live_dir.mkdir(exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    live_chain_path = live_dir / f"{ticker}_chain_{stamp}.parquet"
    live_meta_path = live_dir / f"{ticker}_chain_meta_{stamp}.parquet"
    out.to_parquet(live_chain_path)
    pd.DataFrame([{"spot": spot, "q": q, "asof": asof}]).to_parquet(live_meta_path)

    return dict(spot=spot, q=q, asof=asof, chain=out)


def fetch_history(ticker: str, start: str = "2025-01-01", *, refresh: bool = False) -> pd.Series:
    """Read a local daily-close cache, refreshing it only when requested."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    path = CACHE_DIR / f"{safe}_hist.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)[safe]

    if not refresh:
        raise FileNotFoundError(
            f"no local history cache for {ticker}; rerun with refresh=True to download it"
        )

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
    required = {"bid", "ask", "mid", "open_interest", "T"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"chain is missing required columns: {sorted(missing)}")
    numeric = list(required)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)
    d = d[d["bid"] > min_bid]
    d = d[d["ask"] > d["bid"]]
    rel_spread = (d["ask"] - d["bid"]) / d["mid"].clip(lower=1e-6)
    d = d[rel_spread <= max_rel_spread]
    d = d[d["open_interest"] >= min_oi]
    d = d[(d["T"] >= 5 / 365) & (d["T"] <= 2.0)]
    return d.reset_index(drop=True)
