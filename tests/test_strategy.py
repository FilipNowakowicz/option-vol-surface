"""Accounting and input contracts for Greek P&L attribution."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import run_straddle


def _series(n: int = 5) -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range("2026-01-05", periods=n)
    return (
        pd.Series(np.linspace(100.0, 104.0, n), index=dates),
        pd.Series(np.linspace(20.0, 19.0, n), index=dates),
    )


def test_straddle_attribution_reconciles_to_revaluation():
    spy, vix = _series()
    result = run_straddle(spy, vix, entry_date=spy.index[0], holding_days=4)

    assert len(result.attribution) == 4
    assert result.totals["total_pnl"] == pytest.approx(
        result.totals["sum_attribution"] + result.totals["residual"], abs=1e-12
    )
    assert result.daily["pnl"].iloc[1:].sum() == pytest.approx(result.totals["total_pnl"])


def test_straddle_requires_a_complete_holding_window():
    spy, vix = _series(3)
    with pytest.raises(ValueError, match="insufficient observations"):
        run_straddle(spy, vix, entry_date=spy.index[0], holding_days=3)
