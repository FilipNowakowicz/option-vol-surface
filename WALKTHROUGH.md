# Walkthrough — Options Pricer, Vol Surface & Greek Attribution

A reading guide for the code in this directory. Read this once and the
modules should be self-explanatory. Code references use `file.py:NN`
notation so you can jump in directly.

---

## 1. What the system does, in one sentence

Implement Black-Scholes from scratch with every Greek, cross-check with
Monte Carlo, invert option prices into implied volatilities, fit a smile
surface to real SPY listed options, and use the surface to attribute the
realised P&L of a long ATM straddle into delta/gamma/vega/theta
contributions.

Output: six plots (`mc_convergence.png`, `surface_3d.png`,
`smile_slices.png`, `term_structure.png`, `skew.png`, `attribution.png`),
a printed summary of headline numbers, and **131 pytest tests** that
verify correctness of every analytical Greek and the IV round-trip.

---

## 2. The data flow, end to end

```
              ┌──── Phase 1 / Phase 2 — model layer ────┐
              │                                          │
              │  bs.price, bs.delta, …, bs.veta         │  closed-form
              │              ▲                           │
              │              │  FD verification          │
              │  mc.price_european (antithetic + CV) ───►│  cross-check
              └──────────────────────────────────────────┘
                              │
                              ▼
              ┌──── Phase 3 — inversion ────┐
              │  iv.implied_vol             │  Newton-Raphson on vega
              │     │  no-arb filter        │  Brent fallback
              │     ▼                       │
              │  one σ per quote            │
              └─────────────────────────────┘
                              │
                              ▼
              ┌──── Phase 4 — empirics ────┐
              │  data.fetch_chain (SPY)    │  yfinance chain (cached)
              │     │ data.clean_chain     │  quote-quality filter
              │     ▼                      │
              │  surface.build_surface     │  (K, T, mid) → σ̂
              │     │  surface.diagnose    │  smile / term / skew / butterfly
              │     ▼                      │
              │  3D plot + 2D slices       │
              └────────────────────────────┘
                              │
                              ▼
              ┌──── Phase 5 — strategy ────┐
              │  data.fetch_history(SPY)   │  realised S(t)
              │  data.fetch_history(^VIX)  │  realised σ(t) proxy
              │     │                       │
              │     ▼                       │
              │  strategy.run_straddle     │  daily re-mark
              │     │  delta/gamma/vega/   │  P&L decomposition
              │     │  theta + residual    │
              │     ▼                       │
              │  attribution.png           │
              └────────────────────────────┘
```

The three "layers" — model, inversion, empirics — are deliberately
independent. You can use `bs.py` and `iv.py` without ever touching
yfinance; you can run the surface without running the strategy. The only
glue is in `run.py`.

---

## 3. Black-Scholes in one page

The framework, exactly as used in the code:

- **Underlying** follows GBM under the risk-neutral measure:
  `dS = (r − q) S dt + σ S dW`, with continuous dividend yield `q`.
- **European call price**: `C = S·e^{−qT}·N(d₁) − K·e^{−rT}·N(d₂)` with
  `d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)` and `d₂ = d₁ − σ√T`.
- **Put price**: from put-call parity, `P = K·e^{−rT}·N(−d₂) − S·e^{−qT}·N(−d₁)`.
- **Greeks** are partial derivatives of `C` (or `P`). The ones implemented:

  | Greek    | Differentiate     | Same for call & put? |
  | -------- | ----------------- | -------------------- |
  | delta    | ∂V/∂S             | no                   |
  | gamma    | ∂²V/∂S²           | **yes**              |
  | vega     | ∂V/∂σ             | **yes**              |
  | theta    | −∂V/∂T (calendar) | no                   |
  | rho      | ∂V/∂r             | no                   |
  | vanna    | ∂²V/∂S∂σ          | **yes**              |
  | volga    | ∂²V/∂σ²           | **yes**              |
  | charm    | ∂(delta)/∂t       | no                   |
  | veta     | ∂(vega)/∂t        | **yes**              |

- **Sign of theta** convention used here: `theta = ∂V/∂t` (calendar
  time). Since `T = T_exp − t` decreases as `t` advances, `theta =
  −∂V/∂T`. Most calls have **negative theta** (decay).
- **Put-call parity** is the strongest no-arbitrage identity:
  `C − P = S·e^{−qT} − K·e^{−rT}`. The code tests this to ~1e-10.

That is the entire model. Everything in `bs.py` is one of these formulas.

---

## 4. Module by module

Modules listed in dependency order — later ones build on earlier ones.

### `src/bs.py` — closed-form pricer + every Greek

`bs.price` is six lines (`bs.py:45`). The Greeks are all derived
analytically by hand from the BS formula and the chain rule, then
verified in `tests/test_bs.py` against central finite-difference. Two
small subtleties:

- **`_d1_d2`** (`bs.py:38`) is the shared helper. Computing it once per
  call avoids subtle numerical drift between Greeks that nominally
  should agree (e.g. gamma and vanna both contain `φ(d₁)`).
- **`theta` sign convention** (`bs.py:78`): the standard textbook
  formula is `∂V/∂t`, not `∂V/∂T`. The two differ by a sign. The
  finite-difference verifier in `test_bs.py:test_theta_fd` mirrors this
  by computing `(V(T−h) − V(T+h)) / (2h)`.

`bs.all_greeks` (`bs.py:136`) returns a single dict — convenient for the
straddle attribution because it lets us compute every Greek for both
legs in one pass.

### `src/mc.py` — Monte Carlo cross-check

`mc.price_european` (`mc.py:47`) does three things stacked:

1. **GBM closed-form one-step simulation.** `S_T = S₀·exp((r−q−σ²/2)T +
   σ√T·Z)`, `Z ∼ N(0,1)`. No path discretisation needed for European
   payoffs.
2. **Antithetic variates.** For each draw `Z`, also include `−Z`. Cuts
   variance roughly in half on symmetric payoffs (call/put around ATM)
   for free. Implemented in `_terminal_prices` (`mc.py:29`).
3. **Control variate.** The discounted terminal underlying `Y = e^{−rT}
   S_T` has known mean `E[Y] = S₀·e^{−qT}`. We compute
   `X' = X − ĉ·(Y − E[Y])` where `ĉ = Cov(X,Y)/Var(Y)`, both estimated
   in-sample (`mc.py:69-74`). On ATM calls this knocks another ~3× off
   the stderr.

Stderr is the in-sample standard deviation of the adjusted payoffs
divided by `√N`. **This is what the test asserts**: MC mean should fall
within ~3 standard errors of the analytical price.

The Asian-option function `price_asian` (`mc.py:88`) is a stretch path —
it has no closed-form benchmark, so the stderr is the entire credibility
statement.

### `src/iv.py` — implied volatility solver

`iv.implied_vol` (`iv.py:48`) inverts `BS(σ) = market_price` for σ. Four
layers, in order:

1. **No-arbitrage gate.** Calls must satisfy
   `max(S e^{−qT} − K e^{−rT}, 0) ≤ C ≤ S e^{−qT}`; puts mirror.
   Anything outside returns `NaN` (`iv.py:62-69`). This is **not** a
   smoothing step — we genuinely refuse to invert quotes that no
   non-negative σ could produce.
2. **Initial guess** via Brenner-Subrahmanyam:
   `σ₀ ≈ √(2π/T) · price / S`. Clamped into `[0.05, 2.0]` to keep vega
   well-defined.
3. **Newton-Raphson** on `f(σ) = BS(σ) − mkt`, with `f'(σ) = vega`. Fast
   when it works (~3-5 iters), but abandons if vega vanishes
   (`iv.py:80`) or if it steps outside `[1e-5, 5]`.
4. **Brent fallback** on the same `f`, bracketed by
   `[1e-5, 5]`. Guaranteed to converge if a sign change exists.

`implied_vol_vec` (`iv.py:106`) is just a Python loop; the inner Newton
is fast enough that vectorising in numpy isn't worth the code
complication.

### `src/data.py` — chain fetch + history fetch + cleaning

Three things this module does:

- `fetch_chain(ticker)` (`data.py:38`) pulls the full option chain at
  every listed expiry on the current trade date, with caching. Spot and
  dividend yield come from `_spot_and_yield` (`data.py:24`), which
  handles yfinance's inconsistent yield reporting (sometimes pct,
  sometimes decimal).
- `fetch_history(ticker, start)` (`data.py:110`) is the daily-close
  fetcher used for SPY and ^VIX time series in the strategy.
- `clean_chain` (`data.py:128`) is the **quote-quality filter** —
  `bid > $0.05`, `(ask − bid) / mid ≤ 25%`, `OI ≥ 10`,
  `5d ≤ T ≤ 2y`. These thresholds drop about a third of raw quotes
  (9,847 → 6,325 on 2026-05-20). A looser filter lets garbage IVs pollute
  the surface; a tighter one would discard real OTM strikes.

The cache files in `cache/` are gitignored for the historical series but
**committed for the chain snapshot** so the project remains reproducible
at exactly its quoted date.

### `src/surface.py` — empirical IV surface diagnostics

`build_surface` (`surface.py:35`) is one short function: attach
`F = S e^{(r−q)T}`, `k = ln(K/F)`, and `σ̂ = implied_vol(mid)` to each
row, then drop the rows where IV came back `NaN`.

`diagnose` (`surface.py:173`) returns a `SurfaceDiagnostics` dataclass
with five frames:

- `smile` — `(T, k, σ̂, type, K)` per quote, ready for plotting slices.
- `term_structure` — ATM IV per expiry, via `_interp_at_k0`
  (`surface.py:72`). The trick is to **prefer OTM** options on each
  side: calls for `k ≥ 0`, puts for `k < 0`, since OTM mids are the
  cleanest market quotes. Linear-interpolate between the two strikes
  that bracket `k = 0`.
- `skew` — slope of σ̂ on `k` restricted to `|k| ≤ 0.10`, fit per
  expiry by OLS (`surface.py:94`). On SPY this is sharply negative at
  short T and flattens with maturity.
- `butterfly` — for each expiry, find the call closest to +25Δ and the
  put closest to −25Δ, then compute `σ_25C + σ_25P − 2·σ_ATM`. This is
  a model-free measure of smile curvature.
- `arb_violations` — within each expiry, call mid should be monotone
  decreasing in K. Any violation is reported (`surface.py:155`) and
  **not** silently smoothed.

**Why no SVI / no parametric fit?** Two reasons. First, the project is
about *the empirical surface itself*, not about fitting a particular
parametric form. Second, SVI fits are interview-grade only when you can
articulate the no-arbitrage constraints they impose; without that, it's
window-dressing.

### `src/strategy.py` — ATM straddle daily attribution

`_straddle_greeks` (`strategy.py:42`) is the sum of the per-leg Greeks
of a long call + long put at the same strike. Gamma, vega and veta are
type-agnostic so they're `2 × bs.gamma(...)` etc.; delta and theta need
explicit `call + put`.

`run_straddle` (`strategy.py:54`) walks day by day:

```
for t in [t₀+1, …, t₀+holding_days]:
    sigma(t) = VIX(t)/100         ← realised IV proxy
    S(t)     = SPY(t)
    T(t)     = T(t-1) − Δdays/365
    re-mark every leg at the new (S, σ, T)
    attribute total = δ·ΔS + ½γ·(ΔS)² + ν·Δσ + θ·Δt + residual
```

Greeks for the attribution come from the **prior** day (`prev_greeks`,
`strategy.py:103`). This matches the convention that the first-order
Taylor expansion uses derivatives evaluated at the *start* of the
interval. Using day-t Greeks instead would compute the implicit
expansion around the wrong point.

The output dict at the end includes both `residual_pct_of_total` and
`residual_pct_of_gross` (`strategy.py:130`). The former blows up when
total P&L is near zero; the latter is the meaningful number in those
cases. The success criterion in the README is stated against gross for
that reason.

### `run.py` — orchestrator

Top-down:

1. Print one set of Greeks for an ATM 1y call and verify parity to
   1e-10 (`run.py:main`).
2. Run `mc.price_european` at a path-count grid `[1k, 4k, 16k, 64k,
   256k, 1M]`, plot price ± 2·stderr.
3. Pull SPY chain → clean → build surface → diagnose → plot four
   surface charts (3D, smile slices, term structure, skew).
4. Pull SPY + VIX history → simulate ATM straddle 21 days from the
   most-recent-minus-21 entry date → print the attribution → plot it.

The orchestrator is **deliberately printable** — every section ends in
a one-line summary so you can read the full output of one run and
understand the state of the project at that snapshot date.

---

## 5. Key design decisions, and why

### Why write every Greek closed-form when `py_vollib` exists?

Three reasons:

1. **Interview signal.** "I implemented BS and verified every Greek
   against finite-difference" reads completely differently from "I called
   `py_vollib.greeks`."
2. **Subtle sign conventions.** Theta in particular has at least three
   conventions in the wild (per-year vs per-day, w.r.t. `t` vs `T`,
   continuously-compounded vs simple). Writing it forces you to pick one
   and document it.
3. **Vanna/volga/charm/veta.** These aren't in every library. They show
   up in second-order P&L attribution (the cross-term residual) and are
   useful for stress-testing a position around an event.

### Why Newton-Raphson with Brent fallback, not pure Brent?

Speed in the typical case. Newton converges quadratically when vega is
well-behaved — ~3-5 iterations to 1e-9 on most ATM quotes. Pure Brent
needs 15-30 evaluations to hit the same tolerance. With ~6,000 quotes
per snapshot, this matters. The Brent fallback only fires on deep-OTM
short-T quotes (~1% of the chain on typical days), where it then
guarantees convergence.

### Why antithetic + control variate, not pure-MC + more paths?

The two combined give ~10× variance reduction, which means equivalent
precision at 1/10 the path count. The control variate has zero extra
runtime cost (`S_T` is already computed for the payoff); antithetic
doubles the path count for free by reusing the random draws. Both are
standard techniques in any QR interview rotation.

### Why log-moneyness `k = ln(K/F)` instead of raw `K` or strike-pct?

Three reasons. (i) `k = 0` is exactly ATM under the BS forward, no
spot-adjustment needed. (ii) Skew slopes in `k`-space are roughly
comparable across expiries — they're vol-points per log-unit-of-strike,
which is the right scale. (iii) Most academic literature on the smile
(notably SVI) is parameterised in `k`, so any future stretch like SVI
slot in without re-tooling.

### Why VIX/100 as the realised-IV proxy, not term-matched historical IVs?

Because yfinance doesn't give us a historical surface time-series. The
options are: (i) freeze IV at entry (loses all vega P&L), (ii) use a
mean-reverting toy IV process (loses realism), (iii) use VIX as a
one-dimensional proxy for the front-end of the surface.

(iii) is the least wrong — VIX is genuinely a market-observed
expected-vol number with daily granularity. The trade-off is that VIX is
30-day forward-looking and our straddle is 60-day at entry, so the
mapping is loose. The README documents this; the residual ends up ~1%
of gross which is small enough to call the attribution honest.

---

## 6. Honest limitations to remember

You will be asked about these — internalise them.

1. **BS-flat-σ is a model.** It ignores the smile/term-structure
   movements that the strategy attribution then absorbs into the residual
   via the VIX-proxy simplification. Real desks use sticky-strike or
   sticky-delta dynamics; we don't model either.
2. **No stochastic vol, no jumps.** The MC pricer is pure GBM. A
   research extension would be Heston (closed-form characteristic
   function, no log-moneyness mismatch) or SVI/SSVI (parametric,
   no-arbitrage-aware).
3. **Single-date surface snapshot, not a time series.** The surface
   diagnostics describe SPY on 2026-05-20 only. Inferences about "the
   SPY skew" generally are anecdotal — you'd want surfaces every day for
   a year to draw distributional conclusions.
4. **American-style early exercise ignored.** Listed SPY options are
   technically American, but early exercise is irrational for OTM/ATM
   strikes without dividends large enough to make the exercise boundary
   bind. Acceptable for SPY at ~0.8% yield; would not be acceptable for
   single-name equities with large discrete dividends.
5. **Mid-quote pricing only.** No bid/ask spread handling. The
   strategy P&L assumes you can both buy and sell at mid; in reality
   you'd cross half the spread each way. For SPY ATM options the spread
   is ~1-2% of premium which would chew most of the small P&L in this
   particular window.
6. **yfinance staleness.** Option-chain quote times are not
   strictly synchronous with the spot close. Caveat documented in the
   README.
7. **Continuous dividend yield.** Trailing yfinance yield used as `q`
   throughout. Discrete-dividend forecasting is the right thing for
   single names; not material for SPY.

---

## 7. How to extend

- **Fit SVI to one expiry.** Adds a parametric smile model with a
  no-arbitrage check (calendar + butterfly). Drop-in for
  `surface._skew_per_expiry` consumers. Half-day of work.
- **Add Heston MC.** Path-discretised stochastic vol, calibrated to the
  empirical surface by a least-squares fit on a slice. Tests the variance
  reduction more aggressively.
- **Risk-reversal strategy.** Swap `_straddle_greeks` for a 25Δ-call
  long, 25Δ-put short pair. Exposes the attribution to the skew
  directly. The rest of `strategy.py` is unchanged.
- **Dispersion / index-vs-single-name.** Reuse `iv.py` and the surface
  layer on a basket of single names against an index ETF. Sells index
  vol vs buys constituent vol; cleanest place to add value before stat
  arb.

---

## 8. Where to look for which question

- "What's the BS formula here?" → `bs.py:_d1_d2`, `bs.price`.
- "How is theta defined?" → `bs.py:78` plus the section header at the top.
- "How does the MC variance reduction actually work?" → `mc.py:69-74`.
- "What's the IV solver's fallback strategy?" → `iv.py:79-105`.
- "How are bad quotes filtered?" → `data.py:128` (`clean_chain`).
- "How is ATM IV defined when no exact ATM strike exists?" →
  `surface.py:_interp_at_k0`.
- "How is the skew measured?" → `surface.py:_skew_per_expiry` — OLS on
  OTM-preferred quotes within `|k| ≤ 0.10`.
- "How is the attribution residual computed?" → `strategy.py:113-117`.
- "Why is the strategy residual reported as % of gross, not net?" →
  `strategy.py:130-134` plus README §"Phase 5".
- "Where does the orchestrator live?" → `run.py:main`.
- "What does each test actually check?" → the docstring at the top of
  each file in `tests/`.
