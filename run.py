"""End-to-end: BS pricer sanity, MC convergence, IV surface, straddle attribution.

Prints headline numbers and saves generated plots below ignored ``outputs/``.
The committed chain snapshot remains untouched; live data requires ``--refresh``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import bs, mc
from src.data import clean_chain, fetch_chain, fetch_history
from src.strategy import run_straddle
from src.surface import build_surface, diagnose
from src.svi import calibrate_surface, surface_arbitrage_diagnostics

HERE = Path(__file__).resolve().parent
TICKER = "SPY"
R = 0.04
HOLD_DAYS = 21
DAYS_TO_EXPIRY = 60


def header(s: str):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def mc_convergence(path: Path, S=100, K=100, T=1.0, r=0.03, sigma=0.20, q=0.0):
    """Plot MC ± 2 stderr vs path count, converging to analytical."""
    analytic = bs.price(S, K, T, r, sigma, q, "call")
    grid = [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000]
    rows = []
    for n in grid:
        res = mc.price_european(S, K, T, r, sigma, q, "call", n_paths=n, seed=42)
        rows.append(dict(n=n, mc=res.price, se=res.stderr))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(df["n"], df["mc"], yerr=2 * df["se"], fmt="o-", capsize=4,
                label="MC ± 2 SE", color="C0")
    ax.axhline(analytic, color="black", lw=1.2, ls="--",
               label=f"BS analytical = {analytic:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Call price")
    ax.set_title("Monte Carlo convergence to Black-Scholes (antithetic + CV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return df, analytic


def plot_surface_3d(surf: pd.DataFrame, path: Path):
    """3D scatter of (moneyness k, T, IV)."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(surf["k"], surf["T"], surf["iv"],
                    c=surf["iv"], cmap="viridis", s=8)
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("years to expiry  T")
    ax.set_zlabel("implied vol  σ")
    ax.set_title(f"{TICKER} implied vol surface")
    fig.colorbar(sc, shrink=0.6, label="IV")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_smile_slices(surf: pd.DataFrame, path: Path, max_slices: int = 6):
    """Selected smile slices (IV vs k) at representative expiries."""
    exps = sorted(surf["T"].unique())
    if len(exps) > max_slices:
        idx = np.linspace(0, len(exps) - 1, max_slices).astype(int)
        chosen = [exps[i] for i in idx]
    else:
        chosen = exps

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, T in enumerate(chosen):
        g = surf[surf["T"] == T].sort_values("k")
        ax.plot(g["k"], g["iv"], "o-", ms=3, lw=1.2,
                color=cmap(i / max(len(chosen) - 1, 1)),
                label=f"T={T:.2f}y")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("implied vol  σ")
    ax.set_title(f"{TICKER} smile slices")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_term_structure(term: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(term["T"], term["atm_iv"], "o-", color="C0")
    ax.set_xlabel("years to expiry")
    ax.set_ylabel("ATM implied vol")
    ax.set_title(f"{TICKER} ATM vol term structure")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_skew(skew: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(skew["T"], skew["skew_dIVdk"], "o-", color="C3")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("years to expiry")
    ax.set_ylabel("dIV / dk near ATM")
    ax.set_title(f"{TICKER} skew (steepness of smile near ATM)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_attribution(attribution: pd.DataFrame, path: Path):
    """Cumulative P&L decomposition over the holding period."""
    cum = attribution[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "residual"]].cumsum()
    cum["total"] = attribution["total"].cumsum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cum.index, cum["total"], color="black", lw=2.0, label="total P&L")
    ax.plot(cum.index, cum["delta_pnl"], label="delta")
    ax.plot(cum.index, cum["gamma_pnl"], label="gamma")
    ax.plot(cum.index, cum["vega_pnl"], label="vega")
    ax.plot(cum.index, cum["theta_pnl"], label="theta")
    ax.plot(cum.index, cum["residual"], color="grey", lw=0.8, ls=":", label="residual")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("cumulative P&L ($)")
    ax.set_title("ATM straddle: cumulative P&L decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(refresh: bool = False):
    output_dir = HERE / "outputs" / ("live" if refresh else "frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    header("Phase 1 — Black-Scholes Greek sanity")
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.03, 0.20, 0.0
    g = bs.all_greeks(S, K, T, r, sigma, q, "call")
    print(f"ATM 1y call (S={S}, K={K}, T={T}, r={r}, σ={sigma}, q={q}):")
    for k_, v_ in g.items():
        print(f"  {k_:8s}  {v_:+.6f}")
    C = bs.price(S, K, T, r, sigma, q, "call")
    P = bs.price(S, K, T, r, sigma, q, "put")
    parity_err = abs((C - P) - (S * np.exp(-q * T) - K * np.exp(-r * T)))
    print(f"  put-call parity error: {parity_err:.2e}")

    header("Phase 2 — Monte Carlo convergence vs BS")
    df_mc, analytic = mc_convergence(output_dir / "mc_convergence.png")
    print(df_mc.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"  BS analytical: {analytic:.5f}")
    print("  → mc_convergence.png saved")

    header("Phase 3 + 4 — Real SPY surface")
    data = fetch_chain(TICKER, refresh=refresh)
    spot, q_yfin, asof, chain = data["spot"], data["q"], data["asof"], data["chain"]
    cleaned = clean_chain(chain)
    print(f"  asof          {asof.date()}")
    print(f"  spot          {spot:.2f}")
    print(f"  dividend q    {q_yfin:.4f}")
    print(f"  raw quotes    {len(chain)}")
    print(f"  after clean   {len(cleaned)}")
    surf = build_surface(cleaned, spot=spot, r=R, q=q_yfin)
    print(f"  after IV      {len(surf)} (drops are no-arb violations)")
    diag = diagnose(surf, spot=spot, r=R, q=q_yfin)
    print(f"  expiries      {len(diag.term_structure)}")
    print(f"  arb violations {len(diag.arb_violations)}")
    print()
    print("Term structure (ATM IV):")
    print(diag.term_structure.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Skew (dIV/dk near ATM):")
    print(diag.skew.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    header("Phase 4b — SVI interpolation validation")
    svi = calibrate_surface(surf, spot=spot, r=R, q=q_yfin)
    svi.to_csv(output_dir / "svi_validation.csv", index=False)
    svi_diag = surface_arbitrage_diagnostics(svi)
    print(f"  calibrated expiries {len(svi)}")
    print(f"  held-out IV RMSE   {svi['holdout_iv_rmse'].mean():.4%} (equal-weighted slices)")
    print(f"  linear IV RMSE     {svi['linear_holdout_iv_rmse'].mean():.4%} (same split)")
    print(f"  held-out price RMSE ${svi['holdout_price_rmse'].mean():.4f}")
    print(f"  grid butterfly flags {svi_diag['butterfly_grid_violations']}")
    print(f"  grid calendar flags  {svi_diag['calendar_grid_violations']}")
    print("  → svi_validation.csv saved (diagnostic flags are not constraints)")

    plot_surface_3d(surf, output_dir / "surface_3d.png")
    plot_smile_slices(surf, output_dir / "smile_slices.png")
    plot_term_structure(diag.term_structure, output_dir / "term_structure.png")
    plot_skew(diag.skew, output_dir / "skew.png")
    print("  → surface_3d.png, smile_slices.png, term_structure.png, skew.png saved")

    header("Phase 5 — ATM straddle P&L decomposition")
    try:
        spy_hist = fetch_history(TICKER, start="2026-01-01", refresh=refresh)
        vix_hist = fetch_history("^VIX", start="2026-01-01", refresh=refresh)
    except FileNotFoundError:
        # History is deliberately an ignored local cache: it is not part of
        # the immutable option-chain evidence. Keep the reproducible surface
        # workflow successful rather than making an old illustrative chart a
        # hidden network requirement.
        print("  skipped: local SPY/VIX history is absent; rerun with --refresh for the illustrative attribution")
        header("Done.")
        return
    # Open 21 trading days before the last available bar so we have a clean
    # holding window with realised data on every day.
    entry = spy_hist.index[-(HOLD_DAYS + 1)]
    res = run_straddle(
        spy_hist, vix_hist,
        entry_date=entry,
        days_to_expiry=DAYS_TO_EXPIRY,
        holding_days=HOLD_DAYS,
        r=R, q=q_yfin,
    )
    print(f"  entry         {entry.date()}  S={res.totals['entry_S']:.2f}  σ={res.totals['entry_sigma']:.4f}")
    print(f"  exit          (+{HOLD_DAYS}td)  S={res.totals['exit_S']:.2f}  σ={res.totals['exit_sigma']:.4f}")
    print(f"  premium       ${res.totals['entry_premium']:.2f}")
    print(f"  exit value    ${res.totals['exit_value']:.2f}")
    print(f"  total P&L     ${res.totals['total_pnl']:+.4f}")
    print()
    print(f"  delta P&L     ${res.totals['delta_pnl']:+.4f}")
    print(f"  gamma P&L     ${res.totals['gamma_pnl']:+.4f}")
    print(f"  vega  P&L     ${res.totals['vega_pnl']:+.4f}")
    print(f"  theta P&L     ${res.totals['theta_pnl']:+.4f}")
    print(f"  residual      ${res.totals['residual']:+.4f}"
          f"   ({100 * res.totals['residual_pct_of_gross']:.2f}% of gross |Σ Greeks|)")
    plot_attribution(res.attribution, output_dir / "attribution.png")
    print("  → attribution.png saved")

    header("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download live inputs into ignored runtime caches; never overwrites the committed chain snapshot",
    )
    main(refresh=parser.parse_args().refresh)
