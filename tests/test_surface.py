"""Surface diagnostics should expose, rather than repair, bad quotes."""

import pandas as pd

from src.surface import diagnose


def _surface() -> pd.DataFrame:
    expiry = pd.Timestamp("2026-06-19")
    # The call at K=110 rises in price and breaches its upper bound; the put
    # at K=110 falls in price.  Both should remain visible to diagnostics.
    return pd.DataFrame(
        [
            {"T": 0.5, "expiry": expiry, "type": "call", "strike": 90.0, "mid": 12.0, "k": -0.1, "iv": 0.25},
            {"T": 0.5, "expiry": expiry, "type": "call", "strike": 110.0, "mid": 105.0, "k": 0.1, "iv": 0.22},
            {"T": 0.5, "expiry": expiry, "type": "put", "strike": 90.0, "mid": 3.0, "k": -0.1, "iv": 0.27},
            {"T": 0.5, "expiry": expiry, "type": "put", "strike": 110.0, "mid": 2.0, "k": 0.1, "iv": 0.24},
        ]
    )


def test_diagnose_reports_bounds_and_vertical_spread_violations():
    diagnostics = diagnose(_surface(), spot=100.0, r=0.0, q=0.0)
    violations = diagnostics.arb_violations

    assert {"upper_bound", "vertical_monotonicity"}.issubset(set(violations["check"]))
    assert len(violations) >= 3
    assert {"call", "put"}.issubset(set(violations["type"]))
