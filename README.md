# options-vol-surface — Options Pricer, Implied Vol & Greek Attribution

A from-scratch Black-Scholes engine, Monte Carlo cross-check, implied-volatility
extraction with no-arbitrage filtering, an empirical vol surface from a frozen
SPY option-chain snapshot, and an ATM-straddle P&L decomposed into delta,
gamma, vega, and theta contributions. Built as an educational derivatives
engineering study: every Greek is closed-form and finite-difference verified,
with no black-box pricing library calls.

136 pytest tests cover pricing, IV inversion, quote cleaning, surface
diagnostics, and P&L-attribution accounting.

For the **implementation walkthrough** (data flow, module-by-module, design
decisions, code refs), see [`WALKTHROUGH.md`](WALKTHROUGH.md).

---

## What is here

| Module           | What it does                                                                       |
| ---------------- | ---------------------------------------------------------------------------------- |
| `src/bs.py`      | Closed-form BS price + every first- and second-order Greek                         |
| `src/mc.py`      | GBM Monte Carlo with antithetic + control-variate variance reduction               |
| `src/iv.py`      | Implied vol via Newton-Raphson on vega, with Brent fallback and no-arb filter      |
| `src/data.py`    | yfinance SPY chain + cached history, quote-quality cleaning                        |
| `src/surface.py` | Build IV surface, term structure, skew, butterfly, arbitrage-violation check       |
| `src/strategy.py`| ATM straddle daily re-mark + delta/gamma/vega/theta attribution                    |
| `run.py`         | Reproduces the study from frozen/local inputs; explicit `--refresh` for live data  |
| `tests/`         | Greeks vs finite-difference, MC convergence to BS, IV round-trip, edge cases       |

---

## Headline results (snapshot 2026-05-20, SPY = 741.26)

### Phase 1 — closed-form Greeks
Every analytical Greek matches central finite-difference to ≤ 1e-4 relative
error across 7 grid points × call/put. Put-call parity recovers to 1e-10.

### Phase 2 — Monte Carlo cross-check
For the ATM 1y call (BS = 9.4134), MC converges as:

| paths     | MC ± 2 SE         |
| --------: | :---------------- |
|     1,000 | 9.0772 ± 0.346    |
|     4,000 | 9.3804 ± 0.189    |
|    16,000 | 9.3878 ± 0.095    |
|    64,000 | 9.4191 ± 0.047    |
|   256,000 | 9.4284 ± 0.023    |
| 1,000,000 | 9.4172 ± 0.012    |

Each interval covers the analytical value. Antithetic + control variate
gives roughly 4× variance reduction over raw GBM in the test cases — both
verified in `tests/test_mc.py`.

![mc convergence](mc_convergence.png)

### Phase 3 — Implied vol
Round-trip σ → price → IV recovers σ to ≤ 1e-6 across calls/puts at
ITM/ATM/OTM strikes. Quotes outside the no-arbitrage band return `NaN`
instead of being smoothed. Deep-OTM short-T quotes exercise the Brent
fallback when Newton's vega vanishes.

### Phase 4 — Real SPY vol surface

- Raw chain: **9,847** quotes over 32 expiries
- After quality filter (bid > $0.05, rel-spread ≤ 25%, OI ≥ 10, 5d ≤ T ≤ 2y): **6,325**
- After IV extraction (drops no-arb violations): **6,250** across 27 expiries
- Arbitrage flags (European bounds or vertical monotonicity; not smoothed): **45**

**Term structure (contango):** ATM IV rises monotonically from **13.0% at 1
week** to **19.4% at 1.5 years** — normal upward-sloping shape with no
near-term event premium today.

**Skew (puts richer than calls):** classic SPY shape — extreme at short
expiries and flattening with maturity:

| expiry    | dIV/dk near ATM |
| --------- | --------------: |
| 1 week    | **-1.39**       |
| 1 month   | -0.70           |
| 6 months  | -0.45           |
| 1.5 years | -0.30           |

**Smile curvature (25-delta butterfly):** positive across all expiries, in
the 0.5–2.5 vol-points range, consistent with a tail-fat distribution.

![surface 3d](surface_3d.png)
![smile slices](smile_slices.png)
![term structure](term_structure.png)
![skew](skew.png)

### Phase 5 — ATM straddle P&L attribution

Long ATM straddle opened **2026-04-21** at SPY = 704.08, σ_entry = VIX/100 =
19.50%, T₀ = 60d. Held 21 trading days. Marked daily with σ(t) = VIX(t)/100
as the realised-IV proxy (documented simplification: a true historical surface
time-series is not available from yfinance).

| Leg          | P&L ($)    |
| ------------ | ---------: |
| **Total**    | **+0.12**  |
| Delta        | +10.24     |
| Gamma        |  +3.79     |
| Vega         |  −4.81     |
| Theta        |  −9.98     |
| Residual     |  +0.87     |

Residual = **1.14% of gross |Σ Greek P&Ls|** — within the < 5% success target.

The near-zero net P&L is the operational lesson: SPY rallied +37 points
(delta + gamma earned **+$14**) but VIX dropped 19.5 → 17.4% (vega cost
**$4.8**) and 21 days of decay cost another **$10**. Long-straddle P&L
needs realised vol > implied vol — neither directional rally alone, nor
crush-resistant vega, was sufficient.

![attribution](attribution.png)

---

## How to run

```bash
uv sync
uv run python run.py        # uses frozen/local inputs; writes ignored outputs/frozen/
uv run pytest tests/ -q     # 136 tests
```

`uv` installs the Python version and exact dependencies recorded in `uv.lock`
on Linux, macOS, and Windows.

The committed `cache/SPY_chain*.parquet` files are immutable evidence for the
results above. A normal run always reads them and never contacts yfinance for a
replacement. To compare against live data, run:

```bash
uv run python run.py --refresh
```

This writes a date-stamped option-chain capture under ignored `cache/live/`,
refreshes the ignored local SPY/VIX history cache, and saves comparison figures
under `outputs/live/`; normal runs use `outputs/frozen/`. Neither mode
overwrites the committed snapshot or figures.
The straddle section is an **illustrative Greek
attribution**, not a tradable strategy or backtest. It uses daily closes and
VIX as a 30-day IV proxy, does not model executable option quotes, and should
not be interpreted as evidence of alpha or expected returns.

---

## Limitations (honest)

- **No stochastic vol, no jumps**: BS-flat-σ assumption. Vega P&L attribution
  uses VIX as a one-dimensional proxy for the entire surface, which absorbs
  any smile/term-structure movement into the residual.
- **Single-date surface snapshot, not a time series**: surface diagnostics
  are valid for one trade date only; rebuilding daily would need a paid data
  source.
- **Mid-quote pricing only**: no bid/ask spread cost in the attribution example.
- **yfinance staleness**: option chain quote times are not strictly synchronous
  with the spot close.
- **American-style ignored**: listed SPY options are technically American
  but rarely exercised early; treated as European throughout.
- **Continuous dividend yield**: trailing yfinance yield used as `q`; no
  discrete-dividend forecasting.

These are the right places to be honest about — at interview, "where would
this model break?" is a guaranteed question.

---

## What this demonstrates

- Implementing Black-Scholes and **all** Greeks from scratch and verifying
  every one against finite-difference is the most direct way to show a QR
  interviewer that the mathematics is internalised, not memorised.
- The IV solver's no-arb filter + Brent fallback is the kind of robustness
  layer a research desk actually cares about — naive Newton blows up on
  illiquid strikes.
- The smile/skew/term-structure analysis recovers well-known stylised facts
  (steep front-month skew, contango term structure, persistent butterfly
  curvature) without parametric smoothing — the surface is the data.
- The straddle attribution is the link between **Greeks as derivatives of
  V** and **Greeks as a P&L decomposition**, which is the single thing
  options-trading-house interviews probe most directly.
