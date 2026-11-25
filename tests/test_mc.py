"""MC convergence tests.

For a given seed and n_paths, MC price should fall within ~3 standard errors
of the analytical BS price. This is a *probabilistic* test — fixing the seed
makes it deterministic.

Also: enabling the control variate must produce strictly smaller stderr than
disabling it (the whole point of variance reduction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bs, mc  # noqa: E402


CASES = [
    dict(S=100, K=100, T=1.0, r=0.03, sigma=0.20, q=0.0, option="call"),
    dict(S=100, K=110, T=0.5, r=0.05, sigma=0.30, q=0.02, option="put"),
    dict(S=100, K=90, T=0.25, r=0.01, sigma=0.40, q=0.00, option="call"),
]


@pytest.mark.parametrize("case", CASES)
def test_mc_converges_to_analytical(case):
    analytic = bs.price(**case)
    result = mc.price_european(**case, n_paths=200_000, seed=42)
    z = (result.price - analytic) / result.stderr
    assert abs(z) < 3.5, (
        f"MC {result.price:.4f} ± {result.stderr:.4f}, BS {analytic:.4f}, z={z:.2f}"
    )


@pytest.mark.parametrize("case", CASES)
def test_control_variate_reduces_variance(case):
    no_cv = mc.price_european(**case, n_paths=50_000, seed=7, control_variate=False)
    with_cv = mc.price_european(**case, n_paths=50_000, seed=7, control_variate=True)
    assert with_cv.stderr < no_cv.stderr, (
        f"CV stderr {with_cv.stderr:.5f} >= no-CV stderr {no_cv.stderr:.5f}"
    )


@pytest.mark.parametrize("case", CASES)
def test_stderr_matches_spread_across_seeds(case):
    """The reported stderr must match how much the price actually varies
    between independent runs. Treating antithetic paths as independent
    would break this."""
    import numpy as np

    runs = [mc.price_european(**case, n_paths=2_000, seed=s) for s in range(400)]
    empirical = np.std([r.price for r in runs], ddof=1)
    reported = np.mean([r.stderr for r in runs])
    assert 0.85 < reported / empirical < 1.15, (reported, empirical)
