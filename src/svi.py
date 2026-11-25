"""Per-expiry raw-SVI calibration with deterministic held-out validation.

The five-parameter raw SVI form models total implied variance,

``w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))``.

This module fits each expiry independently.  It enforces positive total
variance through a reparameterisation of ``a`` but does *not* enforce the
full Gatheral--Jacquier butterfly conditions or calendar consistency.  The
reported price-convexity checks are therefore diagnostics, not a certificate
that a fitted surface is arbitrage-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import bs


@dataclass(frozen=True)
class SVIFit:
    """One independently calibrated SVI slice and its validation metrics."""

    T: float
    expiry: pd.Timestamp
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    n_train: int
    n_holdout: int
    train_iv_rmse: float
    holdout_iv_rmse: float
    holdout_iv_mae: float
    linear_holdout_iv_rmse: float
    holdout_price_rmse: float
    total_variance_floor: float
    convexity_violations: int
    success: bool


def total_variance(k: np.ndarray | float, a: float, b: float, rho: float, m: float, sigma: float):
    """Raw-SVI total variance at log-forward-moneyness ``k``."""
    x = np.asarray(k, dtype=float) - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def _otm_quotes(slice_: pd.DataFrame) -> pd.DataFrame:
    """Return one liquid, OTM-preferred quote per strike in stable order."""
    preferred = slice_[
        ((slice_["k"] < 0.0) & (slice_["type"] == "put"))
        | ((slice_["k"] >= 0.0) & (slice_["type"] == "call"))
    ].copy()
    if preferred.empty:
        preferred = slice_.copy()
    # A chain can contain duplicate contract rows.  Fixed sorting makes both
    # their selection and the following train/holdout split reproducible.
    preferred = preferred.sort_values(["strike", "mid", "type"], kind="stable")
    return preferred.drop_duplicates("strike", keep="first").reset_index(drop=True)


def deterministic_split(quotes: pd.DataFrame, holdout_every: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every fifth interior strike is holdout; endpoints remain in training.

    The split is deterministic, based only on sorted strike rank, and keeps
    the fitted domain bracketed by training quotes.  It is validation of
    interpolation within a single snapshot, not a temporal backtest.
    """
    if holdout_every < 2:
        raise ValueError("holdout_every must be at least 2")
    ordered = quotes.sort_values("strike", kind="stable").reset_index(drop=True)
    rank = np.arange(len(ordered))
    holdout = (rank % holdout_every == holdout_every - 1) & (rank > 0) & (rank < len(ordered) - 1)
    return ordered.loc[~holdout].reset_index(drop=True), ordered.loc[holdout].reset_index(drop=True)


def _params(x: np.ndarray) -> tuple[float, float, float, float, float]:
    """Map bounded optimizer values to raw SVI with w(k) >= 1e-10."""
    floor, b, rho, m, sigma = x
    a = floor - b * sigma * np.sqrt(1.0 - rho * rho)
    return float(a), float(b), float(rho), float(m), float(sigma)


def butterfly_g(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral--Jacquier butterfly function; ``g(k) >= 0`` rules out butterfly arbitrage."""
    x = np.asarray(k, dtype=float) - m
    root = np.sqrt(x * x + sigma * sigma)
    w = a + b * (rho * x + root)
    w1 = b * (rho + x / root)
    w2 = b * sigma * sigma / root ** 3
    k = np.asarray(k, dtype=float)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


def _starts(level: float, grid: bool) -> list[list[float]]:
    if not grid:
        return [
            [level * 0.25, 0.10, -0.40, 0.0, 0.15],
            [level * 0.50, 0.25, -0.70, -0.05, 0.25],
        ]
    # Fixed 18-point grid (docs/SVI_REFIT_PLAN.md, variant C).
    return [
        [level * 0.5, 0.20, rho, m, sig]
        for m in (-0.10, 0.0, 0.10)
        for sig in (0.05, 0.15, 0.40)
        for rho in (-0.70, -0.30)
    ]


def _fit_slice(
    train: pd.DataFrame,
    objective: str = "variance",
    grid_starts: bool = False,
    butterfly_weight: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Fit one slice.

    The defaults reproduce the original fit (variant A): unweighted residuals
    in total variance from two starts. ``objective="iv"`` fits implied vols
    instead, ``grid_starts`` uses the fixed 18-start grid and
    ``butterfly_weight`` penalises ``g(k) < 0`` on the training k range.
    """
    k = train["k"].to_numpy(float)
    T = float(train["T"].iloc[0])
    iv = train["iv"].to_numpy(float)
    observed = iv ** 2 * T
    level = max(float(np.median(observed)), 1e-6)
    lower = np.array([1e-10, 1e-8, -0.999, -2.0, 1e-4])
    upper = np.array([2.0, 5.0, 0.999, 2.0, 5.0])
    k_check = np.linspace(k.min(), k.max(), 41)

    def residual(x: np.ndarray) -> np.ndarray:
        params = _params(x)
        w = total_variance(k, *params)
        if objective == "variance":
            out = w - observed
        elif objective == "iv":
            out = np.sqrt(np.maximum(w, 1e-12) / T) - iv
        else:
            raise ValueError(objective)
        if butterfly_weight > 0.0:
            g = butterfly_g(k_check, *params)
            out = np.concatenate([out, butterfly_weight * np.minimum(g, 0.0)])
        return out

    best = None
    for start in _starts(level, grid_starts):
        x0 = np.clip(np.asarray(start), lower + 1e-12, upper - 1e-12)
        result = least_squares(residual, x0=x0, bounds=(lower, upper), max_nfev=1_000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    assert best is not None
    return best.x, bool(best.success)


def _convexity_violations(
    fit: SVIFit, k_min: float, k_max: float, spot: float, r: float, q: float, points: int = 121
) -> int:
    """Count sampled call-price convexity failures over the observed k range."""
    k = np.linspace(k_min, k_max, points)
    variance = total_variance(k, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
    vols = np.sqrt(np.maximum(variance / fit.T, 1e-12))
    forward = spot * np.exp((r - q) * fit.T)
    strikes = forward * np.exp(k)
    calls = bs.price(spot, strikes, fit.T, r, vols, q, "call")
    slopes = np.diff(calls) / np.diff(strikes)
    # Convex call prices require slopes to be non-decreasing.  This is a
    # discrete diagnostic on a finite grid, not an arbitrage proof.
    return int(np.count_nonzero(np.diff(slopes) < -1e-7))


def calibrate_slice(
    slice_: pd.DataFrame, spot: float, r: float, q: float, train_every: int = 1, **fit_options
) -> SVIFit | None:
    """Fit one expiry, returning ``None`` when it lacks a usable holdout set.

    ``train_every > 1`` keeps only every n-th training strike (endpoints kept)
    for the sparse-quote test; the holdout strikes are unchanged.
    """
    quotes = _otm_quotes(slice_)
    train, holdout = deterministic_split(quotes)
    if train_every > 1:
        rank = np.arange(len(train))
        keep = (rank % train_every == 0) | (rank == len(train) - 1)
        train = train.loc[keep].reset_index(drop=True)
    if len(train) < 5 or holdout.empty:
        return None
    x, success = _fit_slice(train, **fit_options)
    a, b, rho, m, sigma = _params(x)
    T = float(train["T"].iloc[0])
    pred_train = np.sqrt(total_variance(train["k"].to_numpy(), a, b, rho, m, sigma) / T)
    pred_holdout = np.sqrt(total_variance(holdout["k"].to_numpy(), a, b, rho, m, sigma) / T)
    train_error = pred_train - train["iv"].to_numpy()
    holdout_error = pred_holdout - holdout["iv"].to_numpy()
    # A deliberately simple interpolation benchmark.  Because endpoints are
    # retained in training, it never has to extrapolate the held-out strikes.
    train_k = train["k"].to_numpy()
    train_w = train["iv"].to_numpy() ** 2 * T
    linear_holdout_iv = np.sqrt(np.interp(holdout["k"].to_numpy(), train_k, train_w) / T)
    linear_holdout_error = linear_holdout_iv - holdout["iv"].to_numpy()
    fitted_prices = bs.price(
        spot, holdout["strike"].to_numpy(), T, r, pred_holdout, q, "call"
    )
    # The OTM canonical data contains puts below F. Convert their observed
    # mids to synthetic call prices with European put-call parity so errors
    # are comparable in one price convention.
    observed_prices = holdout["mid"].to_numpy(float).copy()
    put_mask = holdout["type"].to_numpy() == "put"
    K = holdout["strike"].to_numpy(float)
    observed_prices[put_mask] += spot * np.exp(-q * T) - K[put_mask] * np.exp(-r * T)
    provisional = SVIFit(
        T=T, expiry=pd.Timestamp(train["expiry"].iloc[0]), a=a, b=b, rho=rho, m=m, sigma=sigma,
        n_train=len(train), n_holdout=len(holdout),
        train_iv_rmse=float(np.sqrt(np.mean(train_error ** 2))),
        holdout_iv_rmse=float(np.sqrt(np.mean(holdout_error ** 2))),
        holdout_iv_mae=float(np.mean(np.abs(holdout_error))),
        linear_holdout_iv_rmse=float(np.sqrt(np.mean(linear_holdout_error ** 2))),
        holdout_price_rmse=float(np.sqrt(np.mean((fitted_prices - observed_prices) ** 2))),
        total_variance_floor=float(x[0]), convexity_violations=0, success=success,
    )
    return SVIFit(**{**asdict(provisional), "convexity_violations": _convexity_violations(
        provisional, float(quotes["k"].min()), float(quotes["k"].max()), spot, r, q
    )})


def calibrate_surface(surf: pd.DataFrame, spot: float, r: float, q: float, **options) -> pd.DataFrame:
    """Calibrate independent SVI slices and return reproducible validation rows."""
    fits = []
    for _, slice_ in surf.groupby("expiry", sort=True):
        fit = calibrate_slice(slice_, spot, r, q, **options)
        if fit is not None:
            fits.append(asdict(fit))
    columns = list(SVIFit.__dataclass_fields__)
    return pd.DataFrame(fits, columns=columns).sort_values("T").reset_index(drop=True)


def surface_arbitrage_diagnostics(fits: pd.DataFrame, k_min: float = -0.35, k_max: float = 0.35) -> dict[str, int]:
    """Grid diagnostics for independent fits; they are intentionally not constraints.

    Raw-SVI positive total variance is enforced per slice.  ``butterfly`` is
    the sampled call-convexity count already recorded per slice.  ``calendar``
    counts adjacent-expiry grid points where total variance decreases with T.
    Neither scan establishes global static-arbitrage freedom.
    """
    if fits.empty:
        return {"butterfly_grid_violations": 0, "calendar_grid_violations": 0}
    grid = np.linspace(k_min, k_max, 121)
    ordered = fits.sort_values("T")
    values = np.array([
        total_variance(grid, row.a, row.b, row.rho, row.m, row.sigma)
        for row in ordered.itertuples(index=False)
    ])
    calendar = int(np.count_nonzero(np.diff(values, axis=0) < -1e-7))
    return {
        "butterfly_grid_violations": int(ordered["convexity_violations"].sum()),
        "calendar_grid_violations": calendar,
    }
