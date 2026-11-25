"""Descriptive (not pre-registered): compare held-out errors with bid-ask half-spreads.

Uses the primary refit (variant C) and linear interpolation in total
variance on the same split as test 1 of docs/SVI_REFIT_PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from run import R, TICKER
from src import svi
from src.data import clean_chain, fetch_chain
from src.iv import implied_vol
from src.surface import build_surface


def main() -> None:
    d = fetch_chain(TICKER, refresh=False)
    S, q = d["spot"], d["q"]
    surf = build_surface(clean_chain(d["chain"]), spot=S, r=R, q=q)
    rows = []
    for _, sl in surf.groupby("expiry", sort=True):
        train, hold = svi.deterministic_split(svi._otm_quotes(sl))
        if len(train) < 5 or hold.empty:
            continue
        T = float(train["T"].iloc[0])
        x, _ = svi._fit_slice(train, objective="iv", grid_starts=True)
        fit = np.sqrt(svi.total_variance(hold["k"].to_numpy(), *svi._params(x)) / T)
        lin = np.sqrt(np.interp(hold["k"], train["k"], train["iv"] ** 2 * T) / T)
        half = []
        for _, row in hold.iterrows():
            try:
                lo = implied_vol(row["bid"], S, row["strike"], T, R, q, row["type"])
                hi = implied_vol(row["ask"], S, row["strike"], T, R, q, row["type"])
                half.append((hi - lo) / 2)
            except Exception:
                half.append(np.nan)
        half = np.array(half, float)
        err_svi, err_lin = np.abs(fit - hold["iv"]), np.abs(lin - hold["iv"])
        rows.append(dict(
            T=T, n=len(hold),
            svi_rmse=100 * float(np.sqrt(np.mean(err_svi ** 2))),
            linear_rmse=100 * float(np.sqrt(np.mean(err_lin ** 2))),
            median_half_spread=100 * float(np.nanmedian(half)),
            svi_inside=float(np.nanmean(err_svi <= half)),
            linear_inside=float(np.nanmean(err_lin <= half)),
        ))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    print(f"held-out quotes inside bid-ask: SVI {np.average(df.svi_inside, weights=df.n):.0%}, "
          f"linear {np.average(df.linear_inside, weights=df.n):.0%}")


if __name__ == "__main__":
    main()
