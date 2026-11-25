# SVI refit: plan fixed before running

Written and committed before any of the variants below were run on the
frozen SPY snapshot.

## Why the current fit may be handicapped

`_fit_slice` minimises unweighted squared error in **total variance**
`w = σ²T`, but the reported metric is **implied-vol** RMSE. Total variance
is largest in the wings and at long expiries, so the objective spends most
of its effort there and lets short-dated smiles drift in vol terms. Two
deterministic starts may also leave some slices in poor local minima
(5 of 27 optimiser runs currently report non-convergence).

## Variants (fixed list)

| Id | Change |
| --- | --- |
| A | Current fit (baseline, unchanged) |
| B | Residuals in implied-vol space: `sqrt(w(k)/T) − iv` |
| C | B plus a fixed grid of 18 starts (3 values of `m` × 3 of `σ` × 2 of `ρ`) |
| D | C plus a penalty on the Gatheral–Jacquier butterfly function `g(k) < 0` on the observed `k` range |

**C is the primary variant.** B and D are reported for attribution only.
No other variants are tried on this snapshot.

## Tests

1. **Same split as now** (every fifth interior strike held out, endpoints
   in training). Metric: equal-weighted mean held-out IV RMSE across
   slices, against linear interpolation in total variance on the same
   split (currently 0.122 vol points).
2. **Sparse quotes.** Keep only every third training strike (endpoints
   kept), hold out the same strikes as in test 1. This asks how each method
   does when quotes are sparse, which is where a parametric smile is meant
   to help. Same metric and baseline.

## Decision rule

- C "beats interpolation" on a test only if its mean held-out IV RMSE is
  lower **and** it wins on a majority of slices.
- All four variants and both tests are reported, whatever the outcome.
- If C does not beat interpolation on test 1, the README says so, as it
  does now.

## Amendment before running

- Variant D's penalty weight is fixed at 1.0 (residuals in vol units,
  `g` evaluated at 41 points on the training `k` range).
- Test 1 keeps the slice set the baseline uses (27 expiries). Test 2 drops
  any slice with fewer than 5 training strikes after thinning; the count is
  reported.
